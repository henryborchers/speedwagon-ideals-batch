"""Generates IDEALS batch upload packages.

This Speedwagon plugin creates a GUI interface that accepts a csv file and a
directory of deposit files and transforms them into AD-Items format accepted
by the IDEALS batch ingest system.

A confirmation screen asks users to verify or update mappings from provided
csv headings to established metadata vales for the IDEALS ingest process.

  Typical usage example:

  Input: csv file with file metadata and a directory with data files
  Output: a directory with csv of standardized metadata and nested directories
   of data files and their associated licenses
"""
import abc
import copy
import csv
import os
import shutil
from typing import List, Any, Dict, Sequence, Generic, Optional, Final, Callable

import speedwagon
import speedwagon.workflow
from speedwagon.frontend.interaction import UserRequestFactory, DataItem, AbstractTableEditData
from speedwagon.tasks import TaskBuilder, Result
import pandas as pd


def build_form_data(results,
                    pretask_results
                    ) -> list[Sequence[DataItem]]:
    """Creates DataItems used in table-like displays that support drop-down select style cells.

    Args:
        results: The initial user input, which are not used here.
         Example: {"Input": "/home/user/metadata.csv"}. Elsewhere, results are user_args(???)
        pretask_results: Array of outputs from one or more Subtasks called by
         BatchIngesterWorkflow.initial_task. Example pretask_results[0].data

    Returns: List of tuples. List members represent rows of interface table data and tuple members represent cell data.

    """
    rows: List[Sequence[DataItem]] = []
    mappings = pretask_results[0].data
    # 1
    # def filter_map_only(data):
    #     if data.source == MapMetadata:
    #         return True
    #     return False
    # filter(filter_map_only, pretask_results)
    #
    #
    # # 2
    # filter(lambda x: x.source == MapMetadata, pretask_results)
    #
    # # 3
    # [x for x in pretask_results if x.source == MapMetadata]
    #

    # could try to get the source from each of the results then access via the class name [item.source for item where source = classname]
    # filter
    base_ideals_meta_select = DataItem(
        name='',
        value='',
        editable=True,
        possible_values=mappings.ideals_metadata.get_required() + mappings.ideals_metadata.get_optional() + ['']
    )

    for ideals_meta, input_meta in list(mappings.matches.items()):
        input_select = DataItem(
            name=input_meta,
            value=input_meta,
            editable=True,
            possible_values=list(mappings.matches.values()) + ['']
        )

        ideals_select = copy.deepcopy(base_ideals_meta_select)
        ideals_select.name = ideals_meta
        ideals_select.value = ideals_meta

        rows.append(
            (
                ideals_select,
                input_select,
            )
        )

    # add rows for any unused input column headings
    for input_meta in mappings.unused_input:
        input_select = DataItem(
            name=input_meta,
            value=input_meta,
            editable=True,
            possible_values=[input_meta for input_meta in mappings.unused_input]
        )

        # ideals_select = base_ideals_meta_select
        ideals_select = copy.deepcopy(base_ideals_meta_select)

        rows.append((ideals_select, input_select))
    return rows


class BatchIngesterWorkflow(speedwagon.Workflow):
    """Workflow for generating IDEALS batch ingest package"""

    name = "IDEALS Batch Ingest Builder"
    description = "Creates files for IDEALS batch ingest"

    # TODO: Make some better comments for the description that helps the user know what this is all about

    def job_options(self) -> List[speedwagon.workflow.AbsOutputOptionDataType]:

        validate_metadata = speedwagon.workflow.BooleanSelect("Disable Validation?")
        # validation.default_value = True # Doesn't seem to do anything
        validate_metadata.value = True

        csv_file = speedwagon.workflow.FileSelectData("CSV metadata file")

        #add contional so that both errors aren't triggered if the first fails
        csv_file.add_validation(speedwagon.validators.ExistsOnFileSystem())
        csv_file.add_validation(speedwagon.validators.IsFile())

        files_dir = speedwagon.workflow.DirectorySelect("Files directory")
        files_dir.add_validation(speedwagon.validators.ExistsOnFileSystem())
        files_dir.add_validation(speedwagon.validators.IsDirectory())

        output_dir = speedwagon.workflow.DirectorySelect("Output directory")
        output_dir.add_validation(speedwagon.validators.ExistsOnFileSystem())
        output_dir.add_validation(speedwagon.validators.IsDirectory())


        #TODO: add filter file if we really want to enforce csv

        return [
            csv_file,
            files_dir,
            output_dir,
            validate_metadata
        ]

    def discover_task_metadata(self, initial_results: List[Any], additional_data: Dict[str, Any], **user_args) -> List[
        dict]:
        """

        Args:
            initial_results: the result from the initial processing, MapMetadata
            additional_data: the result from get_additional_data, dictionary with ideals:input
            **user_args: from the initial user input, keys should be from job_options

        Returns:

        """

        return [
            {
                "initial_mappings": initial_results,
                "files_dir": user_args["Files directory"],
                "output_dir": user_args["Output directory"],
                "user_mappings": additional_data,
                "metadata_file": user_args["CSV metadata file"]
            }
        ]

    def initial_task(  # noqa: B027
            self,
            task_builder: TaskBuilder,
            **user_args
    ) -> None:
        csv_file = user_args['CSV metadata file']
        # input_metadata = ['dc:title', 'dc:identifier:uri', 'dc:random']

        task_builder.add_subtask(MapMetadata(csv_file))
        super().initial_task(task_builder, **user_args)

    def get_additional_info(self, user_request_factory: UserRequestFactory, options: dict,
                            pretask_results: list) -> dict:
        def process_data(
                return_data: List[Sequence[DataItem]]
        ) -> dict:
            return {
                row[0].value: row[1].value
                for row in return_data
            }

        mappings_editor = user_request_factory.table_data_editor(
            enter_data=build_form_data,
            process_data=process_data
        )
        mappings_editor.title = "Metadata Map"
        mappings_editor.column_names = ["IDEALS Metadata", "Input Metadata (from CSV file)"]
        user_confirmed_mappings = mappings_editor.get_user_response(options, pretask_results)
        print(user_confirmed_mappings)
        return user_confirmed_mappings

    def create_new_task(self, task_builder: TaskBuilder, **job_args) -> None:
        super().create_new_task(task_builder, **job_args)

        manifest = CreateManifest(user_mappings=job_args["user_mappings"], metadata_file=job_args["metadata_file"],
                                  output_dir=job_args["output_dir"])

        # directory = BuildDirectory(files_dir=job_args["files_dir"], output_dir=job_args["output_dir"])

        task_builder.add_subtask(manifest)

    @classmethod
    def generate_report(cls, results: List[Result], **user_args) -> Optional[str]:
        """
        Args: results: index of all the results from all of the tasks accessed by index number **user_args: the
        initial user input, e.g.  the csv file , the files directory the output director and validate?

        Returns:

        """

        return """Some information about what happened and any validation errors
        """





def get_csv_headers(csv_file) -> list[str]:
    """Extracts column headers from csv input.

    These represent the metadata fields from the user's input csv file.

    Args:
        csv_file: a csv file

    Returns:
        list of headers

    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        return next(reader)


class IdealsMetadataRepresentation:
    """Representation of the IDEALS data schema

      Examples:
          ideals_data.get_optional() # returns the optional metadata fields as a list of strings
          ideals_data.append_metadata(required = ["dc:new_field"]) # add additional required metadata field to existing
    """
    # TODO: add acceptable data types/validation
    required = List[str]
    optional = List[str]

    def get_optional(self) -> list[str]:
        """Get the optional metadata fields

        Returns: List of strings
        """
        return self.optional

    def get_required(self) -> list[str]:
        """Get the required metadata fields

        Returns: List of strings
        """
        return self.required

    def get_metadata(self) -> dict[str:str]:
        """Get all the metadata fields

        Returns: Dictionary with all metadata fields
        """
        return {'required': self.get_required(), 'optional': self.get_optional()}

    def set_metadata(self, required: Optional[list] = None, optional: Optional[list] = None):
        """Overwrite any existing values

        Args:
            required: list of the required metadata fields
            optional: list of the optional metadata fields

        Returns:
            None

        """
        if required is not None:
            self.required = required
        if optional is not None:
            self.optional = optional

    def append_metadata(self, required: Optional[list] = None, optional: Optional[list] = None):
        """Adds additional values to existing metadata

        Args:
            required: list of required metadata fields to add
            optional: list of optional metadata fields to add

        Returns:
            None

        """
        if required is not None:
            self.required = self.required + required
        if optional is not None:
            self.optional = self.optional + optional


class Mapping:
    """Creates a mapping representation between input metadata fields and expected metadata fields. Accepts an optional
    IdealsMetadataRepresentation class input for expected metadata fields.

    Will attempt basic fuzzy matching to pair terms that are incomplete by checking if A appears as a substring in B,
     e.g., "dc:title" will match "title". Monitors which input values have not been paired, which  expected values have
     not been paired, and if all required values have been paired.

     Attributes:
         ideals_metadata: IdealsMetadataRepresentation for the IDEALS data schema
         unused_input: list of the unmapped input fields
         unused_ideals: list fo the unmapped expected fields
         matches: dictionary of the matches, {expected:input}

      Examples:
           my_mapping.map() # performs the matching of inputs and outputs
           my_mapping.requirements_met() # returns True if all expected metadata that are required have been paired
    """

    default_metadata = {
        'required': ['dc:title', 'dc:description', 'dc:subject', 'license'],
        'optional': ['dc:identifier:uri', 'id', 'files', 'file_descriptions']
    }
    matches = {}
    unused_input = []
    unused_ideals = []

    def __init__(self, input_metadata: List,
                 ideals_metadata_representation: Optional[IdealsMetadataRepresentation] = None):

        if ideals_metadata_representation is not None:
            self.ideals_metadata = ideals_metadata_representation
        else:
            self.ideals_metadata = IdealsMetadataRepresentation()
            self.ideals_metadata.set_metadata(required=self.default_metadata['required'],
                                              optional=self.default_metadata['optional'])
        self.unused_ideals = self.ideals_metadata.get_required() + self.ideals_metadata.get_optional()
        self.unused_input = input_metadata

    def match(self, input_val):
        """Matches an input value to an expected value

        Args:
            input_val: a string to match against expected

        Returns: the expected value that matched or None

        """
        match = None

        # first see if there is an exact match
        for ideals_val in self.unused_ideals[:]:
            if input_val == ideals_val:
                match = ideals_val
                self.unused_ideals.remove(ideals_val)
                return match

        # then see if there is anything close
        for ideals_val in self.unused_ideals[:]:
            if input_val in ideals_val or ideals_val in input_val:
                match = ideals_val
                self.unused_ideals.remove(ideals_val)
                break
        return match

    def map(self):
        """Performs the matching for all the input values and updates values that haven't been matched

        Returns:
            None

        """

        for input_val in self.unused_input[:]:
            match = self.match(input_val)
            if match is not None:
                self.matches[match] = input_val
                self.unused_input.remove(input_val)

    def requirements_met(self) -> bool:
        """Tests if all the expected required metadata fields have been matched

        Returns: Bool

        """
        unused_requirements = list(set(self.unused_ideals).intersection(self.ideals_metadata.get_required()))

        if len(unused_requirements) > 0:
            return False
        else:
            return True

    def get_mappings(self) -> dict[str:str]:
        """Get the results matched input and expected pairs

        Returns: a dictionary of expected:input matches

        """
        return self.matches


class MapMetadata(speedwagon.tasks.Subtask):
    """Subtask that matches input metadata fields to expected

      Attributes:
          input_metadata: should be the headers from the user's input csv
          mappings: a Mapping object with mappings from the user's input csv

      Examples:
          mapping_task.work() # performs the mapping and
    """

    def __init__(self, csv_file) -> None:
        super().__init__()
        self.input_metadata = get_csv_headers(csv_file)
        self.mappings = Mapping(self.input_metadata)

    def work(self) -> bool:
        """Sets the mapping object as class _result attribute

        Returns: True if all requirements in the data schema have mapped to inputs

        """
        self.mappings.map()

        self.set_results(self.mappings)
        return self.mappings.requirements_met()  # returning false doesn't seem to have an effect

class BuildDirectory(speedwagon.tasks.Subtask):

    def __init__(self, output_dir, files_dir,
                 manifest):  # manifest will be a Manifest object in the validation workflow, for now a dataframe
        """"""
        super().__init__()
        self.output_dir = output_dir
        self.files_dir = files_dir
        self.manifest = manifest

    def work(self) -> bool:
        # iterate over the manifest and for each row copy the file from the files_dir to output_dir/item{i}/file and
        # copy the license to output_dir/item{i}/license.txt

        for index, row in self.manifest:
            input_file = self.files_dir + row["file"]
            output_file = f"{self.files_dir}/item{index}/{input_file}"
            shutil.copyfile(input_file, output_file)

            file_license = row["license"]

            with open(f"{self.files_dir}/item{index}/license.txt", "w") as out_file:
                out_file.write(file_license)



        return True


class CreateManifest(speedwagon.tasks.Subtask):  # can just be a function and return a Manifest object to BuildDirectory

    def __init__(self, user_mappings, metadata_file, output_dir):
        super().__init__()
        self.user_mappings = user_mappings
        self.metadata_file = metadata_file
        self.output_dir = output_dir

    def work(self) -> bool:

        errors = []
        input_to_ideals_map = {value: key for key, value in self.user_mappings.items()}
        df = pd.read_csv(self.metadata_file)
        df = df[list(self.user_mappings.values())]
        df.rename(columns=input_to_ideals_map, inplace=True)
        errors.append("some error")
        location = self.output_dir + "/batch_manifest.csv"
        print(location)
        df.to_csv(location, index=False)
        self.set_results(
            {
                "location": location,
                "validation_errors": errors
            }
        )

        return True



    # self.log is the logging inside any of the Subtasks, makes most sense inside of work
