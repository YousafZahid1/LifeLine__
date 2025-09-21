from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import uuid
import asyncio
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from enum import Enum

app = FastAPI(title="Social Justice Enhanced API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IncidentType(str, Enum):
    HARASSMENT = "harassment"
    DISCRIMINATION = "discrimination"
    UNSAFE_AREA = "unsafe_area"
    EMERGENCY = "emergency"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCESSIBILITY_ISSUE = "accessibility_issue"
    BIAS_INCIDENT = "bias_incident"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BiasType(str, Enum):
    RACIAL = "racial"
    GENDER = "gender"
    SOCIOECONOMIC = "socioeconomic"
    AGE = "age"
    DISABILITY = "disability"
    LOCATION = "location"

class SafetyIncident(BaseModel):
    id: str
    type: IncidentType
    location: str
    description: str
    severity: SeverityLevel
    timestamp: datetime
    reported_by: str
    verified: bool = False
    coordinates: Optional[Dict[str, float]] = None
    image_data: Optional[str] = None
    bias_detected: bool = False
    bias_types: List[BiasType] = []
    community_impact_score: float = 0.0

class BiasAnalysis(BaseModel):
    id: str
    text: str
    bias_score: float
    bias_types: List[BiasType]
    recommendations: List[str]
    timestamp: datetime
    confidence: float

class CommunityInsight(BaseModel):
    id: str
    type: str
    title: str
    description: str
    impact_level: str
    timestamp: datetime
    data_points: int
    confidence: float

class SafetyRecommendation(BaseModel):
    id: str
    title: str
    description: str
    category: str
    priority: str
    actionable: bool
    estimated_impact: float
    implementation_cost: str

incidents_db: List[SafetyIncident] = []
bias_analyses_db: List[BiasAnalysis] = []
community_insights_db: List[CommunityInsight] = []
safety_recommendations_db: List[SafetyRecommendation] = []

class SocialJusticeAI:
    def __init__(self):
        self.bias_keywords = {
            BiasType.RACIAL: [
                "suspicious person", "looks dangerous", "foreign", "different",
                "sketchy", "thug", "gang member", "illegal"
            ],
            BiasType.GENDER: [
                "women shouldn't", "men are", "typical girl", "typical boy",
                "act like a man", "be a lady", "hysterical", "aggressive"
            ],
            BiasType.SOCIOECONOMIC: [
                "poor area", "bad neighborhood", "ghetto", "rich people",
                "low income", "affluent", "slums", "wealthy"
            ],
            BiasType.AGE: [
                "old people", "young kids", "teenagers are", "elderly",
                "millennials", "boomers", "kids these days", "senior"
            ],
            BiasType.DISABILITY: [
                "disabled person", "handicapped", "wheelchair bound",
                "mentally ill", "crazy", "retarded", "special needs"
            ],
            BiasType.LOCATION: [
                "downtown", "bad part of town", "sketchy area",
                "dangerous neighborhood", "rough area", "ghetto"
            ]
        }
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_trained = False
        
    def detect_bias(self, text: str) -> Dict[str, Any]:
        """Detect bias in text using keyword matching and ML"""
        text_lower = text.lower()
        detected_bias_types = []
        
        for bias_type, keywords in self.bias_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_bias_types.append(bias_type)
        
        bias_score = min(1.0, len(detected_bias_types) * 0.25)
        confidence = 0.8 if detected_bias_types else 0.9
        
        recommendations = self.generate_bias_recommendations(detected_bias_types)
        
        return {
            "bias_types": detected_bias_types,
            "bias_score": bias_score,
            "confidence": confidence,
            "recommendations": recommendations
        }
    
    def generate_bias_recommendations(self, bias_types: List[BiasType]) -> List[str]:
        """Generate recommendations based on detected bias types"""
        recommendations = []
        
        for bias_type in bias_types:
            if bias_type == BiasType.RACIAL:
                recommendations.extend([
                    "Consider using neutral language when describing people",
                    "Focus on behavior rather than appearance or ethnicity"
                ])
            elif bias_type == BiasType.GENDER:
                recommendations.extend([
                    "Use inclusive language that doesn't assume gender roles",
                    "Avoid gender stereotypes in descriptions"
                ])
            elif bias_type == BiasType.SOCIOECONOMIC:
                recommendations.extend([
                    "Avoid assumptions based on neighborhood or economic status",
                    "Focus on specific incidents rather than area generalizations"
                ])
            elif bias_type == BiasType.AGE:
                recommendations.extend([
                    "Avoid age-based assumptions or stereotypes",
                    "Focus on individual behavior rather than age groups"
                ])
            elif bias_type == BiasType.DISABILITY:
                recommendations.extend([
                    "Use person-first language (person with disability)",
                    "Avoid outdated or offensive disability terminology"
                ])
            elif bias_type == BiasType.LOCATION:
                recommendations.extend([
                    "Avoid negative generalizations about neighborhoods",
                    "Focus on specific safety concerns rather than area bias"
                ])
        
        if not recommendations:
            recommendations.append("No bias detected - text appears neutral")
        
        return recommendations
    
    def calculate_community_impact(self, incident: SafetyIncident) -> float:
        """Calculate community impact score for an incident"""
        base_score = 0.5
        
        severity_multipliers = {
            SeverityLevel.LOW: 0.2,
            SeverityLevel.MEDIUM: 0.4,
            SeverityLevel.HIGH: 0.7,
            SeverityLevel.CRITICAL: 1.0
        }
        
        type_multipliers = {
            IncidentType.HARASSMENT: 0.6,
            IncidentType.DISCRIMINATION: 0.8,
            IncidentType.UNSAFE_AREA: 0.4,
            IncidentType.EMERGENCY: 1.0,
            IncidentType.SUSPICIOUS_ACTIVITY: 0.3,
            IncidentType.ACCESSIBILITY_ISSUE: 0.5,
            IncidentType.BIAS_INCIDENT: 0.7
        }
        
        severity_mult = severity_multipliers.get(incident.severity, 0.5)
        type_mult = type_multipliers.get(incident.type, 0.5)
        
        if incident.bias_detected:
            base_score += 0.2
        
        return min(1.0, base_score * severity_mult * type_mult)
    
    def generate_community_insights(self) -> List[CommunityInsight]:
        """Generate AI-powered community insights"""
        insights = []
        
        recent_incidents = [i for i in incidents_db if 
                          i.timestamp > datetime.now() - timedelta(days=7)]
        
        if recent_incidents:
            incident_types = [i.type for i in recent_incidents]
            most_common_type = max(set(incident_types), key=incident_types.count)
            
            insights.append(CommunityInsight(
                id=str(uuid.uuid4()),
                type="safety_pattern",
                title=f"Increase in {most_common_type.value.replace('_', ' ').title()} Reports",
                description=f"Pattern detected: {len([i for i in recent_incidents if i.type == most_common_type])} reports in the last 7 days",
                impact_level="medium",
                timestamp=datetime.now(),
                data_points=len(recent_incidents),
                confidence=0.75
            ))
        
        bias_incidents = [i for i in incidents_db if i.bias_detected]
        if bias_incidents:
            insights.append(CommunityInsight(
                id=str(uuid.uuid4()),
                type="bias_detection",
                title="Bias Detection Alert",
                description=f"{len(bias_incidents)} incidents with potential bias detected",
                impact_level="high",
                timestamp=datetime.now(),
                data_points=len(bias_incidents),
                confidence=0.85
            ))
        
        accessibility_incidents = [i for i in incidents_db if i.type == IncidentType.ACCESSIBILITY_ISSUE]
        if accessibility_incidents:
            insights.append(CommunityInsight(
                id=str(uuid.uuid4()),
                type="accessibility",
                title="Accessibility Concerns",
                description=f"{len(accessibility_incidents)} accessibility issues reported",
                impact_level="medium",
                timestamp=datetime.now(),
                data_points=len(accessibility_incidents),
                confidence=0.8
            ))
        
        return insights
    
    def generate_safety_recommendations(self) -> List[SafetyRecommendation]:
        """Generate AI-powered safety recommendations"""
        recommendations = []
        
        recent_incidents = [i for i in incidents_db if 
                          i.timestamp > datetime.now() - timedelta(days=30)]
        
        if recent_incidents:
            high_severity = [i for i in recent_incidents if i.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
            
            if high_severity:
                recommendations.append(SafetyRecommendation(
                    id=str(uuid.uuid4()),
                    title="Increase Emergency Response",
                    description="High-severity incidents require improved emergency response",
                    category="response",
                    priority="urgent",
                    actionable=True,
                    estimated_impact=0.8,
                    implementation_cost="Medium"
                ))
            
            bias_incidents = [i for i in recent_incidents if i.bias_detected]
            if bias_incidents:
                recommendations.append(SafetyRecommendation(
                    id=str(uuid.uuid4()),
                    title="Bias Awareness Training",
                    description="Provide community training on bias awareness and prevention",
                    category="prevention",
                    priority="high",
                    actionable=True,
                    estimated_impact=0.7,
                    implementation_cost="Low"
                ))
            
            accessibility_incidents = [i for i in recent_incidents if i.type == IncidentType.ACCESSIBILITY_ISSUE]
            if accessibility_incidents:
                recommendations.append(SafetyRecommendation(
                    id=str(uuid.uuid4()),
                    title="Accessibility Audit",
                    description="Conduct comprehensive accessibility audit of public spaces",
                    category="accessibility",
                    priority="medium",
                    actionable=True,
                    estimated_impact=0.6,
                    implementation_cost="High"
                ))
        
        return recommendations

social_justice_ai = SocialJusticeAI()

@app.get("/")
async def root():
    return {
        "message": "Social Justice Enhanced API",
        "version": "2.0.0",
        "features": [
            "Bias detection and analysis",
            "Community safety insights",
            "AI-powered recommendations",
            "Real-time incident tracking",
            "Accessibility monitoring"
        ]
    }

@app.post("/incidents/")
async def report_incident(incident: SafetyIncident, background_tasks: BackgroundTasks):
    """Report a new safety incident with AI analysis"""
    incident.id = str(uuid.uuid4())
    incident.timestamp = datetime.now()
    
    bias_analysis = social_justice_ai.detect_bias(incident.description)
    incident.bias_detected = len(bias_analysis["bias_types"]) > 0
    incident.bias_types = bias_analysis["bias_types"]
    incident.community_impact_score = social_justice_ai.calculate_community_impact(incident)
    
    incidents_db.append(incident)
    
    background_tasks.add_task(update_insights_and_recommendations)
    
    return {
        "message": "Incident reported successfully",
        "incident": incident,
        "bias_analysis": bias_analysis
    }

@app.get("/incidents/")
async def get_incidents(
    incident_type: Optional[IncidentType] = None,
    severity: Optional[SeverityLevel] = None,
    limit: int = 50
):
    """Get incidents with optional filtering"""
    filtered_incidents = incidents_db
    
    if incident_type:
        filtered_incidents = [i for i in filtered_incidents if i.type == incident_type]
    
    if severity:
        filtered_incidents = [i for i in filtered_incidents if i.severity == severity]
    
    recent_incidents = sorted(filtered_incidents, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    return {
        "incidents": recent_incidents,
        "total_count": len(filtered_incidents),
        "bias_incidents": len([i for i in filtered_incidents if i.bias_detected])
    }

@app.post("/bias/analyze")
async def analyze_bias(text: str):
    """Analyze text for bias using AI"""
    bias_analysis = social_justice_ai.detect_bias(text)
    
    analysis = BiasAnalysis(
        id=str(uuid.uuid4()),
        text=text,
        bias_score=bias_analysis["bias_score"],
        bias_types=bias_analysis["bias_types"],
        recommendations=bias_analysis["recommendations"],
        timestamp=datetime.now(),
        confidence=bias_analysis["confidence"]
    )
    
    bias_analyses_db.append(analysis)
    
    return {
        "analysis": analysis,
        "message": "Bias analysis completed"
    }

@app.get("/insights/")
async def get_community_insights():
    """Get AI-generated community insights"""
    insights = social_justice_ai.generate_community_insights()
    community_insights_db.extend(insights)
    
    return {
        "insights": insights,
        "total_insights": len(community_insights_db)
    }

@app.get("/recommendations/")
async def get_safety_recommendations():
    """Get AI-generated safety recommendations"""
    recommendations = social_justice_ai.generate_safety_recommendations()
    safety_recommendations_db.extend(recommendations)
    
    return {
        "recommendations": recommendations,
        "total_recommendations": len(safety_recommendations_db)
    }

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get comprehensive analytics summary"""
    total_incidents = len(incidents_db)
    bias_incidents = len([i for i in incidents_db if i.bias_detected])
    recent_incidents = len([i for i in incidents_db if 
                          i.timestamp > datetime.now() - timedelta(days=7)])
    
    incident_types = {}
    for incident in incidents_db:
        incident_types[incident.type.value] = incident_types.get(incident.type.value, 0) + 1
    
    severity_distribution = {}
    for incident in incidents_db:
        severity_distribution[incident.severity.value] = severity_distribution.get(incident.severity.value, 0) + 1
    
    avg_community_impact = sum(i.community_impact_score for i in incidents_db) / max(total_incidents, 1)
    
    return {
        "total_incidents": total_incidents,
        "bias_incidents": bias_incidents,
        "recent_incidents_7d": recent_incidents,
        "incident_types": incident_types,
        "severity_distribution": severity_distribution,
        "average_community_impact": round(avg_community_impact, 3),
        "community_health_score": round(max(0, 1 - avg_community_impact), 3),
        "bias_detection_rate": round(bias_incidents / max(total_incidents, 1), 3)
    }

@app.get("/accessibility/score")
async def get_accessibility_score():
    """Calculate community accessibility score"""
    accessibility_incidents = [i for i in incidents_db if i.type == IncidentType.ACCESSIBILITY_ISSUE]
    
    if not accessibility_incidents:
        return {"accessibility_score": 1.0, "message": "No accessibility issues reported"}
    
    total_accessibility_issues = len(accessibility_incidents)
    resolved_issues = len([i for i in accessibility_incidents if i.verified])
    
    accessibility_score = resolved_issues / max(total_accessibility_issues, 1)
    
    return {
        "accessibility_score": round(accessibility_score, 3),
        "total_issues": total_accessibility_issues,
        "resolved_issues": resolved_issues,
        "pending_issues": total_accessibility_issues - resolved_issues
    }

async def update_insights_and_recommendations():
    """Background task to update insights and recommendations"""
    await asyncio.sleep(1)
    
    new_insights = social_justice_ai.generate_community_insights()
    community_insights_db.extend(new_insights)
    
    new_recommendations = social_justice_ai.generate_safety_recommendations()
    safety_recommendations_db.extend(new_recommendations)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
