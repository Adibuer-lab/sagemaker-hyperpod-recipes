from pydantic import BaseModel, ConfigDict, Field, model_validator


class NemoRunValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    model_type: str
    nodes: int | None = Field(default=None, gt=0)
    ntasks_per_node: int | None = Field(default=None, gt=0)


class NemoTrainerValidator(BaseModel):
    model_config = ConfigDict(extra="allow")

    devices: int | None = Field(default=None, gt=0)
    num_nodes: int | None = Field(default=None, gt=0)


class NemoRecipeValidator(BaseModel):
    """Top-level validator for NeMo 2.0 recipes."""

    model_config = ConfigDict(extra="allow")

    run: NemoRunValidator
    trainer: NemoTrainerValidator | None = None
    entry_script: str | None = None
    script_args: list[dict[str, object]] | None = None

    @model_validator(mode="after")
    def validate_fields(self):
        if self.run.model_type != "nemo2":
            raise ValueError("run.model_type must be 'nemo2' for NeMo recipes")
        if self.entry_script is None or not str(self.entry_script).strip():
            raise ValueError("entry_script must be provided for NeMo recipes")
        if self.script_args is None:
            raise ValueError("script_args must be provided for NeMo recipes")
        return self
