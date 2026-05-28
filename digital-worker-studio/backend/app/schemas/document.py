from pydantic import BaseModel, Field
from typing import List, Optional

class RiskSchema(BaseModel):
    category: str = Field(description="The category of risk (e.g., Legal, Financial, Operational, Timeline)")
    description: str = Field(description="Detailed description of the risk identified in the document")
    severity: str = Field(description="Severity level: LOW, MEDIUM, or HIGH")

class TaskSchema(BaseModel):
    task: str = Field(description="The actionable task or action item extracted from the text")
    assignee: Optional[str] = Field(default="Unassigned", description="The person or entity assigned to the task, if mentioned")
    deadline: Optional[str] = Field(default="None Specified", description="The deadline or date associated with this task")

class DocumentInsightsSchema(BaseModel):
    executive_summary: str = Field(description="A concise, high-level summary of the document's core content, purpose, and outcomes.")
    risks: List[RiskSchema] = Field(default=[], description="A list of critical liabilities, risks, or concerns discovered in the text.")
    tasks_and_deadlines: List[TaskSchema] = Field(default=[], description="Action items, milestones, or explicit tasks extracted from the document.")