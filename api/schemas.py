from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Defines the expected shape of an incoming prediction request.
    The client must send JSON like: {"text": "some message here"}
    """
    text: str = Field(..., min_length=1, description="The email/SMS message to classify")


class PredictionResponse(BaseModel):
    """
    Defines the shape of the response we send back after a prediction.
    """
    original_text: str
    cleaned_text: str
    prediction: str
    confidence: float