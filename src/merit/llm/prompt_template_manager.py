import os
import asyncio
import logging
import re
from string import Template
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field
import importlib

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplateManager:
    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {"system": "system", "user": "user", "assistant": "assistant"},
        metadata={"help": "Mapping from default roles in prompt template files to specific LLM providers' defined roles."}
    )
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(
        init=False,
        default_factory=dict,
        metadata={"help": "A dict from prompt template names to templates."}
    )


    def __post_init__(self) -> None:
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)
        self.templates_dir = os.path.join(package_dir, "templates")
        self._load_templates()


    def _load_templates(self) -> None:
        if not os.path.exists(self.templates_dir):
            logger.error(f"Templates directory '{self.templates_dir}' does not exist.")
            raise FileNotFoundError(f"Templates directory '{self.templates_dir}' does not exist.")

        logger.info(f"Loading templates from directory: {self.templates_dir}")
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]

                try:
                    module_name = f".llm.templates.{script_name}"
                    module = importlib.import_module(module_name, 'merit')

                    if not hasattr(module, "prompt_template"):
                        logger.error(f"Module '{module_name}' does not define a 'prompt_template'.")
                        raise AttributeError(f"Module '{module_name}' does not define a 'prompt_template'.")

                    prompt_template = module.prompt_template
                    logger.debug(f"Loaded template from {module_name}")

                    if isinstance(prompt_template, Template):
                        self.templates[script_name] = prompt_template
                    elif isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list) and all(
                        isinstance(item, dict) and "role" in item and "content" in item for item in prompt_template
                    ):
                        for item in prompt_template:
                            item["role"] = self.role_mapping.get(item["role"], item["role"])
                            item["content"] = item["content"] if isinstance(item["content"], Template) else Template(item["content"])
                        self.templates[script_name] = prompt_template
                    else:
                        raise TypeError(
                            f"Invalid prompt_template format in '{module_name}.py'. Must be a Template or List[Dict]."
                        )

                    logger.debug(f"Successfully loaded template '{script_name}' from '{module_name}.py'.")

                except Exception as e:
                    logger.error(f"Failed to load template from '{module_name}.py': {e}")
                    raise

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        template = self.get_template(name)
        if isinstance(template, Template):
            try:
                result = template.substitute(**kwargs)
                return result
            except KeyError as e:
                raise ValueError(f"Missing variable for template '{name}': {e}")
        elif isinstance(template, list):
            try:
                rendered_list = [
                    {"role": item["role"], "content": item["content"].substitute(**kwargs)}
                    for item in template
                ]
                return rendered_list
            except KeyError as e:
                raise ValueError(f"Missing variable in chat history template '{name}': {e}")

    def list_template_names(self) -> List[str]:
        return list(self.templates.keys())

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        return self.templates[name]

    def is_template_name_valid(self, name: str) -> bool:
        return name in self.templates
