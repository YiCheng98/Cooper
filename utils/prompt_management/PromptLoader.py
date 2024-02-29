import os

prompt_template_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "prompt_templates")

class PromptLoader:
    def __init__(self):
        self.template_cache = {}

    def load_prompt(self, template_name):
        if template_name in self.template_cache.keys():
            return self.template_cache[template_name]
        else:
            with open(os.path.join(prompt_template_dir, template_name + ".txt"), "r") as file:
                template = file.read()
                self.template_cache[template_name] = template
                return template

    def construct_prompt(self, prompt_template, **kwargs):
        template = self.load_prompt(prompt_template)
        return template.format(**kwargs)