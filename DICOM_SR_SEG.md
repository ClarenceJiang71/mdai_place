

## MD.AI Annotation to SR Export Tool
### Export Annotations from MD.AI's platform into DICOM SR Format

Some MD.ai packeges and some other Python packages need to be inputted. And then there is the of class "SR_SEG_Export". You need to have this class to be in a python file that can be accessed from your envionrment. You could then use this class to finish exporting annotation to DICOM SR format. The SR_SEG_Export class could also be found in this notebook: https://colab.research.google.com/drive/1d4I7wZhlRIS2T0nFo6dhmefi5qDHBSkW?usp=sharing

```python
import mdai.SR_SEG_Export

mdai.SR_SEG_EXPORT.SR_SEG_EXPORT.SR_Export(
  annotation_json = 'mdai_staging_project_L1NpnQBv_annotations_2023-06-28-135404.json',
  metadata_json = 'mdai_staging_project_L1NpnQBv_annotations_2023-06-28-165932.json'
)

```


```python
import mdai
import pandas as pd
from datetime import datetime
import os
import requests

import pydicom
from pydicom.filereader import dcmread
from pydicom.dataset import Dataset

# Imports and Parse Some of the DICOM Standard Files
# -----------------------------------------------
class SR_SEG_Export:
  class SR_Export:
    def __init__(self, annotation_json, metadata_json):
      self.Annotation_Json = annotation_json
      self.Metadata_Json = metadata_json

      self.dicom_standards_setups()
      self.dicom_tags_setup()
      self.create_exports()
    
    def dicom_standards_setups(self):

      ctm_json = requests.get('https://raw.githubusercontent.com/innolitics/dicom-standard/master/standard/ciod_to_modules.json').text
      mta_json = requests.get('https://raw.githubusercontent.com/innolitics/dicom-standard/master/standard/module_to_attributes.json').text
      attributes_json = requests.get('https://raw.githubusercontent.com/innolitics/dicom-standard/master/standard/attributes.json').text
      # ciod_to_modules_dataframe
      ctm_df = pd.read_json(ctm_json)
      # module_to_attributes_dataframe
      mta_df = pd.read_json(mta_json)
      # attributes_dataframe
      attributes_df = pd.read_json(attributes_json)

      # Select basic-text-sr modules
      SR_modules_df = ctm_df[ctm_df['ciodId'] == 'basic-text-sr']
      # Select all basic-text-sr attributes
      SR_attributes_df = mta_df[mta_df['moduleId'].isin(SR_modules_df['moduleId'])]

      attribute_to_keyword_map = dict(zip(attributes_df['tag'], attributes_df['keyword']))
      self.keyword_to_VR_map = dict(zip(attributes_df['keyword'], attributes_df['valueRepresentation']))
      attribute_to_type_map = dict(zip(SR_attributes_df['tag'], SR_attributes_df['type']))

      self.keyword_to_type_map = {}
      for attribute in attribute_to_type_map:
        self.keyword_to_type_map[attribute_to_keyword_map[attribute]] = attribute_to_type_map[attribute]

      # Create dicom heirarchy for SR document (modeled after the Standard DICOM Browser)
      # ---------------------------------------------------
      SR_attributes_df.sort_values('path')
      self.dicom_tag_heirarchy = {}
      for _, row in SR_attributes_df.iterrows():
        if row['path'].count(':') == 1:
          self.dicom_tag_heirarchy[attribute_to_keyword_map[row['tag']]] = {}
        else:
          paths = row['path'].split(':')
          #convert all tags in path to tag format
          parents = []
          for parent in paths[1:-1]:
            parent = f'({parent[:4]},{parent[4:]})'.upper()
            parent = attribute_to_keyword_map[parent]
            parents.append(parent)

          child = paths[-1]
          child = f'({child[:4]},{child[4:]})'.upper()
          child = attribute_to_keyword_map[child]

          #get to last tag sequence
          current_sequence = self.dicom_tag_heirarchy[parents[0]]
          for parent in parents[1:]:
            current_sequence = current_sequence[parent]
          current_sequence[child] = {}

      # Dictionary of VR and their corresponding types
      self.typos ={
          'AE': str,
          'AS': str,
          'AT': pydicom.tag.BaseTag,
          'CS': str,
          'DA': str,
          'DS': pydicom.valuerep.DSfloat,
          'DT': str,
          'FL': float,
          'FD': float,
          'IS': pydicom.valuerep.IS,
          'LO': str,
          'LT': str,
          'OB': bytes,
          'OB or OW': bytes,
          'OD': bytes,
          'OF': bytes,
          'OL': bytes,
          'OV': bytes,
          'OW': bytes,
          'PN': pydicom.valuerep.PersonName,
          'SH': str,
          'SL': int,
          'SQ': pydicom.sequence.Sequence,
          'SS': int,
          'ST': str,
          'SV': int,
          'TM': str,
          'UC': str,
          'UI': pydicom.uid.UID,
          'UL': int,
          'UN': bytes,
          'UR': str,
          'US': int,
          'US or SS': int,
          'UT': str,
          'UV': int,
      }

    def dicom_tags_setup(self):
      # Read Imported JSONs
      results = mdai.common_utils.json_to_dataframe(os.getcwd() + '/' + self.Annotation_Json)
      self.metadata = pd.read_json(os.getcwd() + '/' + self.Metadata_Json)

      # Annotations dataframe
      self.annots_df = results['annotations']
      labels = results['labels']
      self.label_name_map = dict(zip(labels.labelId, labels.labelName))
      self.label_scope_map = dict(zip(labels.labelId, labels.scope))

      # Images DICOM Tags dataframe
      tags = []
      for dataset in self.metadata['datasets']:
        tags.extend(dataset['dicomMetadata'])

      # Create organization of study, series, instance UID & dicom tags
      # ----------------------------------------------------------
      self.studies = self.annots_df.StudyInstanceUID.unique()
      self.tags_df = pd.DataFrame.from_dict(tags) # dataframe of study, series, instance UID & dicom tags
      self.dicom_hierarchy = {}
      for tag in tags:
        study_uid = tag['StudyInstanceUID']
        series_uid = tag['SeriesInstanceUID']
        sop_uid = tag['SOPInstanceUID']

        # Check if already seen study_uid yet (avoids key error)
        if study_uid not in self.dicom_hierarchy: # Using study_uid bc rn it's exam level
          self.dicom_hierarchy[study_uid] = []

        # Dicom_heirarchy is a dictionary with study_uid as keys and a list as value
        # each list contains a dictionary with the series_uid as a key and a list of sop_uids as value
        if not any(series_uid in d for d in self.dicom_hierarchy[study_uid]):
          self.dicom_hierarchy[study_uid].append({series_uid:[]})
        for d in self.dicom_hierarchy[study_uid]: #loops through item in dicom_heriarchy list (just the series_uid dict)
          if series_uid in d:
            d[series_uid].append(sop_uid)

    # Helper functions to place DICOM tags into SR document Template
    # ---------------------------------------------------
    '''
    > Iterates through a given sequence of tags from the standard DICOM heirarchy
    > Checks if the tag exists in the current DICOM file's headers
    >>  If it does then it adds the tag to the SR document dataset
    > Recursively calls itself to add tags in sequences and
    >>  Checks if a sequence contains all its required tags and adds them if so
    > Returns the SR document dataset with all tags added
    > If there were no tags added then returns False
    '''
    def place_tags(self, curr_dataset, curr_seq):
      sequences = {}
      added = False
      # Iterate through sequence to add tags and find sequences
      for keyword in curr_seq:
        if keyword in self.dicom_tags:
          curr_dataset = self.add_to_dataset(curr_dataset, keyword, self.dicom_tags[keyword], True)
          added = True
        if self.keyword_to_VR_map[keyword] == 'SQ':
          sequences[keyword] = curr_seq[keyword]

      # Iterate through sequences to add tags and recursively search within sequences for tags
      for keyword in sequences:
        if keyword == 'ContentSequence': # Skips ContentSequence since it's meant to contain the annotations data
            continue
        seq = sequences[keyword]
        new_dataset = Dataset()
        new_dataset = self.place_tags(new_dataset, seq)
        if new_dataset:
          if self.keyword_to_VR_map[keyword] == 'SQ':
            new_dataset = [new_dataset] # Pydicom requires sequences to be in a list
          if self.check_required(new_dataset, seq):
            added = True
            curr_dataset = self.add_to_dataset(curr_dataset, keyword, new_dataset, True)

      if added:
        return curr_dataset

      return False

    # Checks if a sequence contains all its required tags
    def check_required(self, curr_dataset, curr_seq):
      for keyword in curr_seq:
        tag_type = self.keyword_to_type_map[keyword]
        if keyword not in curr_dataset and '1' == tag_type:
          return False
      return True

    # Adds tag to dataset and if the tag already exists then
    # Replaces tag if replace=True if not then does nothing
    def add_to_dataset(self, dataset, keyword, value, replace):
      VR = self.keyword_to_VR_map[keyword]

      # If the tag is a sequence then the value in dicom_tags will be a list containing dictionary so need to convert to sequence format
      if type(value) == list and VR == 'SQ':
        if type(value[0]) == dict:
          value = self.dict_to_sequence(value)

      # If the tag is a byte encoding then need to switch it to so from string
      if self.typos[VR] == bytes and value != None:
        value = value[2:-1].encode('UTF-8') # removes b' and '

      # If the tag is an int/float encoding then need to switch it to so from string
      if self.typos[VR] == int or self.typos[VR] == float:
        if value != None:
          value = self.typos[VR](value)

      # check if tag already in dataset
      if keyword in dataset:
        if not replace:
          return dataset
        dataset[keyword].value = value
        return dataset

      if 'or SS' in VR and type(value) == int: # Fix bug when VR == 'US or SS' and the value is negative (it always defaults to US)
        if value < 0:
          VR = 'SS'

      dataset.add_new(keyword, VR, value)
      return dataset

    # Creates a sequence from a list of dictionaries
    def dict_to_sequence(self, dict_seq_list):
      sequences = []
      for dict_seq in dict_seq_list:
        seq = Dataset()
        for keyword in dict_seq:
          if self.keyword_to_VR_map[keyword] == 'SQ':
            inner_seq = self.dict_to_sequence(dict_seq[keyword])
            seq = self.add_to_dataset(seq, keyword, inner_seq, True)
          else:
            seq = self.add_to_dataset(seq, keyword, dict_seq[keyword], True)
        sequences.append(seq)
      return sequences

    def create_exports(self):
      # Iterate through each study and create SR document for each annotator in each study
      # Save output to Output folder
      # ---------------------------------------------------
      try:
        os.mkdir('Output')
      except:
        pass

      from io import BytesIO
      document_file = requests.get('https://github.com/spike-h/SRDocs/raw/main/Simple%20SR%20-%20RSNA.dcm')

      for study_uid in self.studies:
        #load file template
        ds = dcmread(BytesIO(document_file.content))

        self.dicom_tags = self.tags_df[self.tags_df.StudyInstanceUID == study_uid].dicomTags.values[0]
        annotations = self.annots_df[self.annots_df.StudyInstanceUID == study_uid]
        annotators = annotations.createdById.unique()
        series_uid = pydicom.uid.generate_uid(prefix=None)
        instance_uid = pydicom.uid.generate_uid(prefix=None)
        date = datetime.now().strftime('%Y%m%d')
        time = datetime.now().strftime('%H%M%S')

        # Place all the tags from the dicom into the SR document
        ds = self.place_tags(ds, self.dicom_tag_heirarchy)

        # modify file metadata
        ds.file_meta.MediaStorageSOPInstanceUID = instance_uid                              # Media Storage SOP Instance UID
        ds.file_meta.ImplementationClassUID = str(pydicom.uid.PYDICOM_IMPLEMENTATION_UID)   # Implementation Class UID
        ds.file_meta.ImplementationVersionName = str(pydicom.__version__)                   # Implementation Version Name

        # delete tags
        del ds[0x00080012]  # Instance Creation Date
        del ds[0x00080013]  # Instance Creation Time
        del ds[0x00080014]  # Instance Creator UID
        # del ds[0x00100030]  # Patient's Birth Date

        # modify tags
        #-------------------------

        ds['SOPClassUID'].value = '1.2.840.10008.5.1.4.1.1.88.22' # SOP Class UID = enhanced SR storage
        ds[0x00080018].value = instance_uid  # SOPInstanceUID
        ds[0x0008103e].value = str(self.metadata['name'].values[0])  # Series Description
        ds[0x00080021].value = str(date)  # Series Date
        ds[0x00080023].value = str(date)  # Content Date
        ds[0x00080031].value = str(time)  # Series Time
        ds[0x00080033].value = str(time)  # Content Time

        ds[0x00181020].value = ''   # Software Versions

        ds[0x0020000d].value = str(study_uid)   # Study Instance UID
        ds[0x0020000e].value = str(series_uid)   # Series Instance UID
        ds[0x00200011].value = str(1)           # Series Number

        # create dicom hierarchy
        dicom_hier = self.dicom_hierarchy[study_uid]
        series_sequence = []
        for series in dicom_hier:
          for key in series:
            sops = series[key]
            series_hier = Dataset()
            sop_sequence = []
            for sop in sops:
              sop_data = Dataset()
              if 'SOPClassUID' in self.dicom_tags:
                sop_data.ReferencedSOPClassUID = self.dicom_tags['SOPClassUID']
              sop_data.ReferencedSOPInstanceUID = sop
              sop_sequence.append(sop_data)
            series_hier.ReferencedSOPSequence = sop_sequence
            series_hier.SeriesInstanceUID = key
            series_sequence.append(series_hier)

        ds[0x0040a375][0].ReferencedSeriesSequence = series_sequence
        ds[0x0040a375][0].StudyInstanceUID = study_uid

        # add tags
        ds[0x00080005] = pydicom.dataelem.DataElement(0x00080005, 'CS', 'ISO_IR 192')       # Specific Character Set

        # create content for each annotator
        for i in range(len(annotators)):
        
          instance_number = i+1
          ds[0x00200013] = pydicom.dataelem.DataElement(0x00200013, 'IS', str(instance_number)) # Instance Number
          ds[0x0040a730][0][0x0040a123].value = f'Annotator{instance_number}'
          ds[0x0040a078][0][0x0040a123].value = f'Annotator{instance_number}'
          anns = annotations[annotations.createdById == annotators[i]]

          anns_map = {}
          def annotator_iteration(row):
            annotation = []
            label_id = row['labelId']
            parent_id = row['parentLabelId']
            annotation.extend([parent_id, row['scope'], row['SOPInstanceUID'], row['SeriesInstanceUID']])
            if 'SOPClassUID' in self.dicom_tags:
                annotation.append(self.dicom_tags['SOPClassUID'])

            if label_id not in anns_map:
              anns_map[label_id] = []
            anns_map[label_id].append(annotation)

          anns.apply(annotator_iteration, axis=1)

          # annotator_iteration has extraneous labels for those with child labels as it creates 2 separate entries for the child label and the parent label
          for label_id in anns_map:
            for annot in anns_map[label_id]:
              if annot[0] != None:

                if annot[0] not in anns_map: # Fixes edge case where a child label appears with no parent label for that annotator
                  continue                   # Occurs when another annotator adds a child label to a different annotator's label

                for j in range(len(anns_map[annot[0]])-1, -1, -1): #iterate backwards so can delete while iterating
                  parent_annot = anns_map[annot[0]][j]
                  if ((type(parent_annot[2]) == type(annot[2]) and type(annot[2] == float)) and (type(parent_annot[3]) == type(annot[3]) and type(annot[3] == float))) or ((parent_annot[2] == annot[2])  and (parent_annot[3] == annot[3])): # check if series and sop uid are same
                    del anns_map[annot[0]][j]

          content_sequence = []
          code_number = 43770 #hello

          # Create a list of labelIds ordered from exam to series to image
          ordered_labels = []
          j = 0
          for label_id in anns_map:
            if self.label_scope_map[label_id] == 'EXAM':
              ordered_labels.insert(0, label_id)
              j += 1
            elif self.label_scope_map[label_id] == 'INSTANCE':
              ordered_labels.append(label_id)
            else:
              ordered_labels.insert(j, label_id)

          for label_id in ordered_labels:
            for a in anns_map[label_id]:
              # Add 'Referenced Segment' if label is in IMAGE scope
              if a[1] == 'INSTANCE':
                content = Dataset()
                content.ValueType = 'IMAGE'
                referenced_sequence_ds = Dataset()
                if len(a) > 4:
                  referenced_sequence_ds.ReferencedSOPClassUID = a[4]
                referenced_sequence_ds.ReferencedSOPInstanceUID = a[2]
                content.ReferencedSOPSequence = [referenced_sequence_ds]

                code_sequence_ds = Dataset()
                code_sequence_ds.CodeValue = str(code_number)
                code_sequence_ds.CodingSchemeDesignator = '99MDAI'
                code_sequence_ds.CodeMeaning = 'Referenced Image'
                code_sequence = [code_sequence_ds]
                content.ConceptNameCodeSequence = code_sequence
                code_number += 1
                content_sequence.append(content)

              # Add parent label to text value
              content = Dataset()
              code_sequence_ds = Dataset()
              if a[0] != None:
                code_name = self.label_name_map[a[0]]
              else:
                code_name = self.label_name_map[label_id]
              code_sequence_ds.CodeValue = str(hash(code_name))[1:6]
              code_sequence_ds.CodingSchemeDesignator = '99MDAI'
              code_sequence_ds.CodeMeaning = code_name
              code_sequence = [code_sequence_ds]
              content.ConceptNameCodeSequence = code_sequence

              # Add child label text
              text_value = ''
              if a[0] != None:
                text_value = ','.join(map(lambda labelId: self.label_name_map[labelId], [label_id]))
                text_value += '\n'
                content.TextValue = text_value
              # Add 'Series UID:'
              if a[1] == 'SERIES':
                text_value += f'Series UID: {series_uid}'
                content.TextValue = text_value
              if text_value != '':
                content.ValueType = 'TEXT'
              else:
                content.ValueType = 'CONTAINER'
              content_sequence.append(content)

          ds[0x0040a730][1][0x0040a730][0].ContentSequence = content_sequence


          ds.save_as(f'{os.getcwd()}/Output/DICOM_SR_{study_uid}_annotator_{instance_number}.dcm')



```
