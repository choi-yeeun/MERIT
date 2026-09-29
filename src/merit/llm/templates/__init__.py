"""Prompt templates.

Every ``*.py`` file in this directory (except this one) is auto-discovered by
:class:`merit.llm.PromptTemplateManager`; the filename without the extension is
the template name passed to ``render()``. Each file must define a module-level
``prompt_template``.
"""
