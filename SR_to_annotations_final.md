# SR to Annotations/JSON Documentation

This function is used to convert an SR file into an MD.ai Annotation note or JSON format. It is meant to convert the entirety of the SR's **content sequence** into text while keeping note of the referenced studies.

The JSON format will have two fields: "Referenced DICOM" and "SR Content". Referenced DICOM is a list of dictionaries containing a "Study UID" field which points to the referenced studies (indicated by the content sequence's ReferencedSOPSequence tag). SR Content will be a list of lines of text, each containing a field from the SR document.

The annotation note format will import an annotation into the given project and dataset based on the studies that the SR references. The annotation will have a note containing the SR's content. You must initiate an mdai_client for this to work. Additionally, you must go into the UI and create a label, then input the label_id into the function.
______

  Inputs:

    `file_path` - File path to the SR (required)
    `json_out` - Boolean flag to determine if should output to JSON (optional)
    `project_id` & `dataset_id` & `label_id` - Project information necessary to output SR to annotation note. All must be present if any are present. (optional)
    `mdai_client` - mdai client object instantiated by calling `mdai.Client`. Must be present to export SR to annotation note.
Outputs:


  If `json_out` is `True` then there will be a json file in your cwd called "SR_content". If all the project and client information is filled out, then there will be an annotation with the SR content as an annotation note, for each study in the project that is referenced by the SR.


## Example Usage:


```python
import mdai

# Get variables from project info tab and user settings
DOMAIN = 'public.md.ai'
YOUR_PERSONAL_TOKEN = '8b80c4ca0f'
mdai_client = mdai.Client(domain=DOMAIN, access_token=YOUR_PERSONAL_TOKEN)

dataset_id = 'D_0Z4qeN'
project_id = 'L1NpnQBv'
label_id = 'L_QnlPAg'

file_path = 'path_to_SR.dcm'
```

    Successfully authenticated to staging.md.ai.



```python
SR_to_Annot(file_path,
            dataset_id=dataset_id,
            project_id=project_id,
            label_id=label_id,
            mdai_client=mdai_client)
```

    Importing 1 annotations into project L1NpnQBv, dataset D_0Z4qeN...                                  
    Successfully imported 1 / 1 annotations into project L1NpnQBv, dataset D_0Z4qeN.


## Source Code:


```python
import pydicom
from pydicom import dcmread, Dataset
import json

def SR_to_Annot(file_path, json_out=False, project_id='', dataset_id='', label_id='', mdai_client=None):
  ds = dcmread(file_path)

  # Get the referenced Dicom Files
  referenced_dicoms = []
  for study_seq in ds.CurrentRequestedProcedureEvidenceSequence:
    referenced_study = {}
    study_UID = study_seq.StudyInstanceUID
    referenced_study['Study UID'] = study_UID
    referenced_dicoms.append(referenced_study)

  content_seq_list = list(ds.ContentSequence)

  content = []
  run_through(content, content_seq_list)

  final_content = []
  # print('SR CONTENT:')
  # print('-'*10)
  for annot in content:
    annot = list(filter(None, annot))
    final_content.append(" - ".join(annot))
    # print(" - ".join(annot))
  # print('-'*10)

  if json_out:
    out_json = {}
    out_json['Referenced DICOM'] = referenced_dicoms
    out_json['SR Content'] = final_content

    # Serializing json
    json_object = json.dumps(out_json, indent=4)

    # Writing to sample.json
    with open("SR_content.json", "w") as outfile:
        outfile.write(json_object)

  if project_id or dataset_id or label_id or mdai_client:
    if not project_id:
      print('Please add in the "project_id" argument')
    if not dataset_id:
      print('Please add in the "dataset_id" argument')
    if not label_id:
      print('Please add in the "label_id" argument')
    if not mdai_client:
      print('Please add in the "mdai_client" argument')

    annotations = []
    for dicom_dict in referenced_dicoms:
      study_uid = dicom_dict['Study UID']
      note = '\n'.join(final_content)
      annot_dict = {
        'labelId': label_id,
        'StudyInstanceUID': study_uid,
        'note': note
      }
      annotations.append(annot_dict)

    mdai_client.import_annotations(annotations, project_id, dataset_id)


def run_through(content, content_seq_list):

  for content_seq in content_seq_list:
    parent_labels = []
    child_labels = []
    notes = []

    if 'RelationshipType' in content_seq:
      if content_seq.RelationshipType == 'HAS ACQ CONTEXT':
        continue

    if content_seq.ValueType == 'IMAGE':
      if 'ReferencedSOPSequence' in content_seq:
        for ref_seq in content_seq.ReferencedSOPSequence:
          if 'ReferencedSOPClassUID' in ref_seq:
            notes.append(f'\n   Referenced SOP Class UID = {ref_seq.ReferencedSOPClassUID}')
          if 'ReferencedSOPInstanceUID' in ref_seq:
            notes.append(f'\n   Referenced SOP Instance UID = {ref_seq.ReferencedSOPInstanceUID}')
          if 'ReferencedSegmentNumber' in ref_seq:
            notes.append(f'\n   Referenced Segment Number = {ref_seq.ReferencedSegmentNumber}')
      else:
        continue

    if 'ConceptNameCodeSequence' in content_seq:
      if len(content_seq.ConceptNameCodeSequence) > 0:
        parent_labels.append(content_seq.ConceptNameCodeSequence[0].CodeMeaning)
    if 'ConceptCodeSequence' in content_seq:
      if len(content_seq.ConceptCodeSequence) > 0:
        child_labels.append(content_seq.ConceptCodeSequence[0].CodeMeaning)

    if 'DateTime' in content_seq:
      notes.append(content_seq.DateTime)
    if 'Date' in content_seq:
      notes.append(content_seq.Date)
    if 'PersonName' in content_seq:
      notes.append(str(content_seq.PersonName))
    if 'UID' in content_seq:
      notes.append(content_seq.UID)
    if 'TextValue' in content_seq:
      # notes.append(content_seq.TextValue)
      child_labels.append(content_seq.TextValue)
    if 'MeasuredValueSequence' in content_seq:
      if len(content_seq.MeasuredValueSequence) > 0:
        units = content_seq.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue
        notes.append(str(content_seq.MeasuredValueSequence[0].NumericValue) + units)

    if 'ContentSequence' in content_seq:
      run_through(content, list(content_seq.ContentSequence))
    else:
      content.append([', '.join(parent_labels), ', '.join(child_labels), ", ".join(notes)])
```


```python
import mdai

# Get variables from project info tab and user settings
DOMAIN = 'staging.md.ai'
YOUR_PERSONAL_TOKEN = '8b80c4ca0f05876d561a34622395e487'
mdai_client = mdai.Client(domain=DOMAIN, access_token=YOUR_PERSONAL_TOKEN)

dataset_id = 'D_0Z4qeN'
project_id = 'L1NpnQBv'
```

    Successfully authenticated to staging.md.ai.



```python
SR_to_Annot('/content/1.2.826.0.1.3680043.8.498.10302341752378028659650064735258616799 (1).dcm',
            dataset_id=dataset_id,
            project_id=project_id,
            label_id='L_QnlPAg'
            mdai_client=mdai_client)
```

    SR CONTENT:
    ----------
    Country of Language - United States
    Procedure Reported - Imaging Procedure
    Tracking Unique Identifier - 1.2.826.0.1.3680043.8.498.11859752560909707449344997884042315009
    Clinical finding not suspected - Thromboembolic pulmonary hypertension (disorder)
    Importing 1 annotations into project L1NpnQBv, dataset D_0Z4qeN...                                  
    Successfully imported 1 / 1 annotations into project L1NpnQBv, dataset D_0Z4qeN.

