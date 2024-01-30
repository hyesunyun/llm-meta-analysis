# Code is adapted from bigscience-worshop/promptsource
# https://github.com/bigscience-workshop/promptsource/blob/main/promptsource/templates.py

import os
import uuid
import pkg_resources
import yaml
from jinja2 import BaseLoader, Environment
import logging
from typing import Dict, List

env = Environment(loader=BaseLoader)

# Allow the python function zip()
env.globals.update(zip=zip)

# Local path to the folder containing the templates
TEMPLATES_FOLDER_PATH = pkg_resources.resource_filename(__name__, "templates")

class Template(yaml.YAMLObject):
    """
    A prompt template.
    """

    yaml_tag = "!Template"

    def __init__(self, name, prompt, reference):
        """
        Creates a prompt template.

        A prompt template is expressed in Jinja.
        Generally, the prompt should provide information on the desired
        behavior, e.g., text passage and instructions.

        :param name: unique name (per dataset) for template
        :param prompt: template input prompt expressed in Jinja
        :param reference: string describing author or paper reference for template
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.prompt = prompt
        self.reference = reference

    def get_id(self):
        """
        Returns the id of the template

        :return: unique id for template
        """
        return self.id

    def get_name(self):
        """
        Returns the name of the template

        :return: unique (per dataset) name for template
        """
        return self.name

    def get_reference(self):
        """
        Returns the bibliographic reference (or author) for the template

        :return: reference as a string
        """
        return self.reference

    def apply(self, example):
        """
        Creates a prompt by applying this template to an example

        :param example: the dataset example to create a prompt for
        :return: list of strings (if the prompt was split, then the list will have multiple strings)
        """
        jinja = self.prompt

        rtemplate = env.from_string(jinja)

        # Renders the Jinja template
        rendered_example = rtemplate.render(**example)

        return rendered_example

class DatasetTemplates:
    """
    Class that wraps all templates for a specific dataset/subset and implements all the helper
    functions necessary to read the yaml file
    """

    TEMPLATES_KEY = "templates"
    DATASET_KEY = "dataset"
    SUBSET_KEY = "subset"
    TEMPLATE_FILENAME = "templates.yaml"

    def __init__(self, dataset_name: str, subset_name: str = None):
        self.dataset_name: str = dataset_name
        self.subset_name: str = subset_name
        # dictionary is keyed by template name.
        self.templates: Dict = self.read_from_file()

        # Mapping from template name to template id
        self.name_to_id_mapping = {}
        self.sync_mapping()

    def sync_mapping(self) -> None:
        """
        Re-compute the name_to_id_mapping to ensure it is in sync with self.templates
        """
        self.name_to_id_mapping = {template.name: template.id for template in self.templates.values()}

    @property
    def all_template_names(self) -> List[str]:
        """
        Sorted list of all templates names for this dataset
        """
        return sorted([template.name for template in self.templates.values()])

    @property
    def folder_path(self) -> str:
        if self.subset_name:
            return os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name, self.subset_name)
        else:
            return os.path.join(TEMPLATES_FOLDER_PATH, self.dataset_name)

    @property
    def yaml_path(self) -> str:
        return os.path.join(self.folder_path, self.TEMPLATE_FILENAME)

    def format_for_dump(self) -> Dict:
        """
        Create a formatted dictionary for the class attributes
        """
        formatted_dict = {self.DATASET_KEY: self.dataset_name, self.TEMPLATES_KEY: self.templates}
        if self.subset_name:
            formatted_dict[self.SUBSET_KEY] = self.subset_name
        return formatted_dict

    def read_from_file(self) -> Dict:
        """
        Reads a file containing a prompt collection.
        """

        if not os.path.exists(self.yaml_path):
            dataset_name = f"{self.dataset_name} {self.subset_name}" if self.subset_name else self.dataset_name
            logging.warning(
                f"Tried instantiating `DatasetTemplates` for {dataset_name}, but no prompts found. "
                "Please ignore this warning if you are creating new prompts for this dataset."
            )
            return {}
        yaml_dict = yaml.load(open(self.yaml_path, "r"), Loader=yaml.FullLoader)
        return yaml_dict[self.TEMPLATES_KEY]


    def __getitem__(self, template_key: str) -> "Template":
        return self.templates[self.name_to_id_mapping[template_key]]

    def __len__(self) -> int:
        return len(self.templates)