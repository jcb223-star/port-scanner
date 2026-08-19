import joblib
import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from contextlib import asynccontextmanager

# Global dictionary to store the model artifact upon startup
ml_models = {}

# Resolve the model path relative to this file, not the process's current
# working directory, so it loads correctly regardless of where uvicorn is
# launched from.
MODEL_PATH = Path(__file__).parent / "artifacts" / "model_pipeline.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model pipeline during startup to prevent reloading
    on every individual request, optimizing inference latency.
    """
    try:
        ml_models["pipeline"] = joblib.load(MODEL_PATH)
        print(f"Model pipeline successfully loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        ml_models["pipeline"] = None
    yield
    # Clean up the model reference on shutdown if necessary
    ml_models.clear()

# Initialize FastAPI application with the lifespan handler
app = FastAPI(
    title="ML Model Inference API",
    description="Production-ready API for serving scikit-learn & xgboost pipelines",
    version="1.0.0",
    lifespan=lifespan
)

# Define Pydantic schema matching your input feature names and data types
class ModelInput(BaseModel):
    age: float = Field(..., description="Age of the individual")
    salary: float = Field(..., description="Annual salary")
    score_a: float = Field(..., description="Metric score A")
    score_b: float = Field(..., description="Metric score B")
    tenure: float = Field(..., description="Job tenure in years")
    department: str = Field(..., description="Department name")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 29.0,
                "salary": 72000.0,
                "score_a": 0.91,
                "score_b": 1.12,
                "tenure": 2.0,
                "department": "Finance"
            }
        }
    )

class ModelOutput(BaseModel):
    prediction: int = Field(..., description="Predicted class label (0 or 1)")
    # None when the loaded classifier doesn't support predict_proba, so
    # "no probability available" stays distinguishable from a genuine 0.0
    # confidence score.
    probability: Optional[float] = Field(None, description="Prediction probability score, if available")


@app.get("/")
def read_root():
    """Health check endpoint to verify the service is running and the model is loaded."""
    model_loaded = ml_models.get("pipeline") is not None
    return {
        "status": "online",
        "service": "ML Inference Pipeline",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=ModelOutput)
def predict(payload: ModelInput):
    """
    Accepts raw JSON input, packages it into a DataFrame, passes it
    through the end-to-end sklearn pipeline, and returns the prediction.
    """
    pipeline = ml_models.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")

    try:
        # Convert Pydantic model payload into a single-row DataFrame
        input_data = pd.DataFrame([payload.model_dump()])

        # Run inference via the bundled feature engineering & model pipeline
        prediction = int(pipeline.predict(input_data)[0])

        # Check if the classifier supports probability outputs
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            probability = float(pipeline.predict_proba(input_data)[0][1])
        else:
            probability = None

        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


# --- Local Execution Hook ---
if __name__ == "__main__":
    import uvicorn
    # Run the application locally using Uvicorn server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
