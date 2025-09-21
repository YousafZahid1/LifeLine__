# Social Justice Resources API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import uuid
import math
import asyncio
from enum import Enum

app = FastAPI(title="Social Justice Resources API", version="1.0.0")

# Add CORS middleware for SwiftUI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Enums
# -------------------------------
class ResourceType(str, Enum):
    COMMUNITY_CENTER = "community_center"
    LIBRARY = "library"
    PARK = "park"
    HEALTH_CENTER = "health_center"
    SHELTER = "shelter"

class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# -------------------------------
# Data Models
# -------------------------------
class Resource(BaseModel):
    id: str
    name: str
    type: ResourceType
    address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    risk_factor: float = 0.0  # 0.0 (safest) to 1.0 (highest risk)
    average_rating: float = 0.0  # 1.0 to 5.0
    total_ratings: int = 0
    trust_indicators: Dict[str, bool] = {
        "community_verified": False,
        "official_partnership": False,
        "regular_updates": False,
        "transparent_reporting": False
    }
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class Incident(BaseModel):
    id: str
    resource_id: str
    type: str
    description: str
    severity: IncidentSeverity
    timestamp: datetime
    verified: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    reported_by: Optional[str] = None

class EnvironmentalFactor(BaseModel):
    id: str
    resource_id: str
    factor: str  # e.g., "lighting", "maintenance", "accessibility"
    impact: str
    risk_level: float  # 0.0 to 1.0
    timestamp: datetime = datetime.now()

class CommunityComment(BaseModel):
    id: str
    resource_id: str
    author: str
    rating: int  # 1-5
    text: str
    helpful_count: int = 0
    timestamp: datetime = datetime.now()

class RiskAssessmentRequest(BaseModel):
    resource_id: str
    current_time: Optional[datetime] = None
    weather_condition: Optional[str] = None
    crowd_density: Optional[float] = None

# -------------------------------
# In-memory "database"
# -------------------------------
resources_db: List[Resource] = []
incidents_db: List[Incident] = []
environmental_factors_db: List[EnvironmentalFactor] = []
comments_db: List[CommunityComment] = []

# -------------------------------
# AI Risk Calculation Engine
# -------------------------------
class AIRiskCalculator:
    @staticmethod
    def calculate_risk_factor(resource_id: str) -> float:
        """Calculate AI-driven risk factor for a resource"""
        # Get all data for the resource
        resource_incidents = [i for i in incidents_db if i.resource_id == resource_id]
        resource_factors = [f for f in environmental_factors_db if f.resource_id == resource_id]
        resource_comments = [c for c in comments_db if c.resource_id == resource_id]
        
        # Base risk from incidents (weighted by severity and recency)
        incident_risk = 0.0
        for incident in resource_incidents:
            severity_weight = {"low": 0.2, "medium": 0.5, "high": 0.8}[incident.severity]
            time_weight = max(0.1, 1.0 - (datetime.now() - incident.timestamp).days / 30)
            incident_risk += severity_weight * time_weight
        
        # Environmental factors risk
        env_risk = sum(f.risk_level for f in resource_factors) / max(len(resource_factors), 1)
        
        # Community sentiment (inverse of average rating)
        sentiment_risk = 0.0
        if resource_comments:
            avg_rating = sum(c.rating for c in resource_comments) / len(resource_comments)
            sentiment_risk = (5.0 - avg_rating) / 4.0  # Convert to 0-1 scale
        
        # Weighted combination
        total_risk = (
            incident_risk * 0.4 +      # 40% incidents
            env_risk * 0.3 +           # 30% environmental
            sentiment_risk * 0.3       # 30% community sentiment
        )
        
        return min(max(total_risk, 0.0), 1.0)
    
    @staticmethod
    def calculate_trust_score(trust_indicators: Dict[str, bool]) -> float:
        """Calculate trust score from indicators"""
        active_indicators = sum(1 for active in trust_indicators.values() if active)
        return active_indicators / len(trust_indicators)

# -------------------------------
# API Endpoints
# -------------------------------

@app.get("/")
async def root():
    return {"message": "Social Justice Resources API", "version": "1.0.0"}

# Resource Management
@app.post("/resources/")
async def create_resource(resource: Resource):
    resource.id = str(uuid.uuid4())
    resource.created_at = datetime.now()
    resource.updated_at = datetime.now()
    resources_db.append(resource)
    return {"message": "Resource created successfully", "resource": resource}

@app.get("/resources/")
async def get_resources(resource_type: Optional[ResourceType] = None):
    filtered_resources = resources_db
    if resource_type:
        filtered_resources = [r for r in resources_db if r.type == resource_type]
    
    # Calculate current risk factors and ratings
    for resource in filtered_resources:
        resource.risk_factor = AIRiskCalculator.calculate_risk_factor(resource.id)
        resource_comments = [c for c in comments_db if c.resource_id == resource.id]
        if resource_comments:
            resource.average_rating = sum(c.rating for c in resource_comments) / len(resource_comments)
            resource.total_ratings = len(resource_comments)
    
    return {"resources": filtered_resources}

@app.get("/resources/{resource_id}")
async def get_resource(resource_id: str):
    resource = next((r for r in resources_db if r.id == resource_id), None)
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
    
    # Update with current data
    resource.risk_factor = AIRiskCalculator.calculate_risk_factor(resource_id)
    resource_comments = [c for c in comments_db if c.resource_id == resource_id]
    if resource_comments:
        resource.average_rating = sum(c.rating for c in resource_comments) / len(resource_comments)
        resource.total_ratings = len(resource_comments)
    
    return {"resource": resource}

# Incident Management
@app.post("/incidents/")
async def report_incident(incident: Incident):
    incident.id = str(uuid.uuid4())
    incident.timestamp = datetime.now()
    incidents_db.append(incident)
    
    # Update resource risk factor
    resource = next((r for r in resources_db if r.id == incident.resource_id), None)
    if resource:
        resource.risk_factor = AIRiskCalculator.calculate_risk_factor(incident.resource_id)
        resource.updated_at = datetime.now()
    
    return {"message": "Incident reported successfully", "incident": incident}

@app.get("/resources/{resource_id}/incidents")
async def get_resource_incidents(resource_id: str):
    incidents = [i for i in incidents_db if i.resource_id == resource_id]
    return {"incidents": incidents}

# Environmental Factors
@app.post("/environmental-factors/")
async def add_environmental_factor(factor: EnvironmentalFactor):
    factor.id = str(uuid.uuid4())
    factor.timestamp = datetime.now()
    environmental_factors_db.append(factor)
    
    # Update resource risk factor
    resource = next((r for r in resources_db if r.id == factor.resource_id), None)
    if resource:
        resource.risk_factor = AIRiskCalculator.calculate_risk_factor(factor.resource_id)
        resource.updated_at = datetime.now()
    
    return {"message": "Environmental factor added successfully", "factor": factor}

@app.get("/resources/{resource_id}/environmental-factors")
async def get_resource_environmental_factors(resource_id: str):
    factors = [f for f in environmental_factors_db if f.resource_id == resource_id]
    return {"environmental_factors": factors}

# Community Comments
@app.post("/comments/")
async def add_comment(comment: CommunityComment):
    comment.id = str(uuid.uuid4())
    comment.timestamp = datetime.now()
    comments_db.append(comment)
    
    # Update resource rating
    resource = next((r for r in resources_db if r.id == comment.resource_id), None)
    if resource:
        resource_comments = [c for c in comments_db if c.resource_id == comment.resource_id]
        resource.average_rating = sum(c.rating for c in resource_comments) / len(resource_comments)
        resource.total_ratings = len(resource_comments)
        resource.risk_factor = AIRiskCalculator.calculate_risk_factor(comment.resource_id)
        resource.updated_at = datetime.now()
    
    return {"message": "Comment added successfully", "comment": comment}

@app.get("/resources/{resource_id}/comments")
async def get_resource_comments(resource_id: str):
    comments = [c for c in comments_db if c.resource_id == resource_id]
    return {"comments": comments}

@app.post("/comments/{comment_id}/helpful")
async def mark_comment_helpful(comment_id: str):
    comment = next((c for c in comments_db if c.id == comment_id), None)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    
    comment.helpful_count += 1
    return {"message": "Comment marked as helpful", "helpful_count": comment.helpful_count}

# AI Risk Assessment
@app.post("/risk-assessment/")
async def assess_risk(request: RiskAssessmentRequest):
    resource = next((r for r in resources_db if r.id == request.resource_id), None)
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
    
    # Calculate base risk factor
    base_risk = AIRiskCalculator.calculate_risk_factor(request.resource_id)
    
    # Apply real-time adjustments
    time_adjustment = 0.0
    if request.current_time:
        hour = request.current_time.hour
        if 22 <= hour or hour <= 6:  # Night time
            time_adjustment = 0.2
    
    weather_adjustment = 0.0
    if request.weather_condition in ["rain", "storm", "snow"]:
        weather_adjustment = 0.1
    
    crowd_adjustment = 0.0
    if request.crowd_density:
        if request.crowd_density > 0.8:  # Very crowded
            crowd_adjustment = 0.15
        elif request.crowd_density < 0.2:  # Very empty
            crowd_adjustment = 0.1
    
    final_risk = min(max(base_risk + time_adjustment + weather_adjustment + crowd_adjustment, 0.0), 1.0)
    
    return {
        "resource_id": request.resource_id,
        "base_risk_factor": base_risk,
        "real_time_risk_factor": final_risk,
        "adjustments": {
            "time_adjustment": time_adjustment,
            "weather_adjustment": weather_adjustment,
            "crowd_adjustment": crowd_adjustment
        },
        "risk_level": "low" if final_risk <= 0.3 else "medium" if final_risk <= 0.6 else "high"
    }

# Statistics and Analytics
@app.get("/stats/")
async def get_statistics():
    total_resources = len(resources_db)
    total_incidents = len(incidents_db)
    total_comments = len(comments_db)
    
    # Calculate average risk across all resources
    if resources_db:
        avg_risk = sum(AIRiskCalculator.calculate_risk_factor(r.id) for r in resources_db) / len(resources_db)
        safe_resources = sum(1 for r in resources_db if AIRiskCalculator.calculate_risk_factor(r.id) <= 0.3)
    else:
        avg_risk = 0.0
        safe_resources = 0
    
    return {
        "total_resources": total_resources,
        "total_incidents": total_incidents,
        "total_comments": total_comments,
        "average_risk_factor": round(avg_risk, 3),
        "safe_resources": safe_resources,
        "high_risk_resources": total_resources - safe_resources
    }

# Search resources
@app.get("/resources/search")
async def search_resources(q: str):
    """Search resources by name or address"""
    filtered_resources = []
    query = q.lower()
    
    for resource in resources_db:
        if (query in resource.name.lower() or 
            query in resource.address.lower() or
            query in resource.type.value.lower()):
            filtered_resources.append(resource)
    
    # Calculate current risk factors and ratings
    for resource in filtered_resources:
        resource.risk_factor = AIRiskCalculator.calculate_risk_factor(resource.id)
        resource_comments = [c for c in comments_db if c.resource_id == resource.id]
        if resource_comments:
            resource.average_rating = sum(c.rating for c in resource_comments) / len(resource_comments)
            resource.total_ratings = len(resource_comments)
    
    return {"resources": filtered_resources}

# Get resources nearby a location
@app.get("/resources/nearby")
async def get_resources_nearby(latitude: float, longitude: float, radius: float = 5.0):
    """Get resources within a certain radius of a location"""
    # For now, return all resources since we don't have real geolocation data
    # In a real app, you'd calculate distances using the Haversine formula
    nearby_resources = resources_db
    
    # Calculate current risk factors and ratings
    for resource in nearby_resources:
        resource.risk_factor = AIRiskCalculator.calculate_risk_factor(resource.id)
        resource_comments = [c for c in comments_db if c.resource_id == resource.id]
        if resource_comments:
            resource.average_rating = sum(c.rating for c in resource_comments) / len(resource_comments)
            resource.total_ratings = len(resource_comments)
    
    return {"resources": nearby_resources}

# Real-time data processing (background task)
async def process_real_time_data():
    """Process real-time data updates"""
    while True:
        # Update all resource risk factors
        for resource in resources_db:
            old_risk = resource.risk_factor
            resource.risk_factor = AIRiskCalculator.calculate_risk_factor(resource.id)
            resource.updated_at = datetime.now()
        
        await asyncio.sleep(300)  # Update every 5 minutes

# Start background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_real_time_data())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
