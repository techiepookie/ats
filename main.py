
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Any
import httpx
import json
import random
from fastapi import File, UploadFile, Form
import speech_recognition as sr
import tempfile
import os
from gtts import gTTS
import uuid
import io
import time

app = FastAPI()

# Global state for interview tracking
current_question_number = 0
total_questions = 5  # Default number of questions per interview
interview_answers = []  # Store all answers for final evaluation

# Directory for temporary audio files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Replace with your OpenRouter API key
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ElevenLabs API key - you should use an environment variable for this in production
ELEVENLABS_API_KEY = "sk_ac6cbfe4e6ead10b573b61dbe9c83f5d566b6f8ff02766e2"  # Replace with your actual key
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Female HR voice options for ElevenLabs
VOICE_OPTIONS = {
    "interviewer": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah - professional female voice
        "model_id": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "speaking_rate": 0.9  # Slightly slower, more calm speaking rate
    }
}

class InterviewRequest(BaseModel):
    role: str
    difficulty: Optional[str] = "medium"
    duration: Optional[int] = 3  # minutes

class QuestionResponse(BaseModel):
    content: str
    type: str = "question"
    
class VoiceSettings(BaseModel):
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    speaking_rate: float = 1.0

class AnswerSubmission(BaseModel):
    role: str
    question: str
    answer: str

class InterviewMetric(BaseModel):
    score: float
    description: str

class InterviewPerformance(BaseModel):
    pace: InterviewMetric
    tone: InterviewMetric
    clarity: InterviewMetric
    empathy: InterviewMetric
    relevancy: InterviewMetric
    sentiment: InterviewMetric
    confidence: InterviewMetric
    completeness: InterviewMetric
    average_score: float

class WorkStyleMetric(BaseModel):
    overall: float
    description: str
    work_approach: Dict[str, float]
    team_dynamics: Dict[str, float]
    work_ethic: Dict[str, float]

class ProfessionalCompetenciesMetric(BaseModel):
    overall: float
    description: str
    leadership_management: float
    problem_solving_decision_making: float
    communication_skills: float
    cognitive_adaptability: float

class PersonalityCompatibilitiesMetric(BaseModel):
    overall: float
    description: str
    openness_conventionality: Dict[str, float]
    perfectionism_flexibility: Dict[str, float]
    extraversion_introversion: Dict[str, float]
    agreeableness_assertiveness: Dict[str, float]

class CareerGrowthMetric(BaseModel):
    overall: float
    description: str
    ambition_drive: float
    risk_attitude: Dict[str, float]
    career_focus: Dict[str, float]
    learning_orientation: float

class StressManagementMetric(BaseModel):
    overall: float
    description: str
    stress_response: Dict[str, float]
    work_life_balance: float
    pressure_handling: float
    recovery_ability: Dict[str, float]

class CulturalFitMetric(BaseModel):
    overall: float
    description: str
    organizational_culture: float
    cultural_sensitivity: float
    social_responsibility: float
    adaptability_structure: Dict[str, float]

class JobCompatibility(BaseModel):
    professional_competencies: ProfessionalCompetenciesMetric
    work_style_ethics: WorkStyleMetric
    personality_compatibility: PersonalityCompatibilitiesMetric
    career_growth: CareerGrowthMetric
    stress_management: StressManagementMetric
    cultural_fit: CulturalFitMetric
    average_score: float

class InterviewResult(BaseModel):
    interview_performance: InterviewPerformance
    job_compatibility: JobCompatibility

# Behavioral and soft skills questions templates
SOFT_SKILLS_TEMPLATES = {
    "conflict_resolution": [
        "Describe a situation where you had to resolve a conflict within your team. How did you handle it?",
        "Tell me about a time when you had a disagreement with a colleague. How did you resolve it?",
        "How do you typically handle conflicts in the workplace?"
    ],
    "leadership": [
        "Describe a situation where you took the lead on a challenging project.",
        "Tell me about a time when you had to motivate a team through a difficult situation.",
        "How do you approach delegating tasks and responsibilities?"
    ],
    "communication": [
        "Describe a situation where you had to explain a complex concept to someone with little background knowledge.",
        "Tell me about a time when you had to deliver difficult feedback to a colleague or team member.",
        "How do you ensure effective communication within a team with diverse backgrounds?"
    ],
    "problem_solving": [
        "Tell me about a complex problem you faced at work and how you approached solving it.",
        "Describe a situation where you had to think creatively to solve a problem.",
        "How do you typically approach troubleshooting and problem-solving in your work?"
    ],
    "adaptability": [
        "Describe a situation where you had to quickly adapt to changes in requirements or circumstances.",
        "Tell me about a time when you had to learn a new skill or technology in a short amount of time.",
        "How do you handle unexpected challenges or setbacks in your work?"
    ],
    "teamwork": [
        "Describe your approach to collaborating with team members who have different working styles.",
        "Tell me about a successful team project and your contribution to its success.",
        "How do you handle situations where team members aren't pulling their weight?"
    ],
    "time_management": [
        "How do you prioritize tasks when you have multiple deadlines approaching?",
        "Describe a situation where you had to manage multiple projects simultaneously.",
        "Tell me about a time when you had to work under tight time constraints."
    ],
    "stress_management": [
        "How do you handle high-pressure situations or tight deadlines?",
        "Describe a stressful situation you faced at work and how you managed it.",
        "What techniques do you use to stay calm and focused under pressure?"
    ]
}

# Role-specific technical question templates
ROLE_TEMPLATES = {
    "developer": [
        "Tell me about your experience with {technology}.",
        "How would you implement {feature} using {language}?",
        "Describe a challenging technical problem you've solved.",
        "How do you approach debugging a complex issue in a {technology} application?",
        "Explain your understanding of {concept} in software development."
    ],
    "management": [
        "How do you handle team conflicts?",
        "Describe your approach to project planning and resource allocation.",
        "Tell me about a time you had to make a difficult decision as a leader.",
        "How do you measure team performance and productivity?",
        "What strategies do you use to keep your team motivated?"
    ],
    "design": [
        "Walk me through your design process.",
        "How do you incorporate user feedback into your designs?",
        "Tell me about a design challenge you faced and how you overcame it.",
        "How do you balance aesthetics with functionality?",
        "What design trends are you currently following and why?"
    ]
}

# Technologies and concepts for developer questions
TECHNOLOGIES = ["JavaScript", "Python", "React", "Node.js", "MongoDB", "SQL", "AWS", "Docker", "Kubernetes", "Git"]
FEATURES = ["authentication system", "data visualization dashboard", "RESTful API", "real-time chat", "payment integration"]
LANGUAGES = ["JavaScript", "Python", "Java", "C#", "TypeScript", "Go"]
CONCEPTS = ["microservices", "serverless architecture", "CI/CD", "test-driven development", "design patterns"]

def generate_question_for_role(role, include_soft_skills=True):
    """Generate a relevant question based on the selected role with soft skills option"""
    role = role.lower()
    
    # Randomly decide whether to ask a technical or soft skills question
    if include_soft_skills and random.random() < 0.6:  # 60% chance for soft skills questions
        # Select a random soft skills category
        soft_skill_category = random.choice(list(SOFT_SKILLS_TEMPLATES.keys()))
        return random.choice(SOFT_SKILLS_TEMPLATES[soft_skill_category])
    
    # Fall back to technical/role-specific question
    if role not in ROLE_TEMPLATES:
        role = "developer"  # Default to developer if role not found
    
    template = random.choice(ROLE_TEMPLATES[role])
    
    # Fill in placeholders for developer questions
    if role == "developer":
        template = template.replace("{technology}", random.choice(TECHNOLOGIES))
        template = template.replace("{feature}", random.choice(FEATURES))
        template = template.replace("{language}", random.choice(LANGUAGES))
        template = template.replace("{concept}", random.choice(CONCEPTS))
    
    return template

async def get_ai_response(messages):
    """Get response from OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000"  # Update with your domain
    }
    
    payload = {
        "model": "deepseek/deepseek-r1:free",  # Try a different model
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 10000
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout
        try:
            response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                # Fall back to a pre-generated response if API fails
                return "I've reviewed your response and would like to explore further. Could you elaborate more on your approach and provide specific examples from your experience?"
        except Exception as e:
            print(f"Exception when calling OpenRouter API: {str(e)}")
            # Fall back to a pre-generated response if API call fails
            return "Thank you for your response. Let's explore a different aspect of your experience. Can you share a situation where you had to adapt to unexpected changes?"

async def text_to_speech_elevenlabs(text, voice_config=None):
    """Convert text to speech using ElevenLabs API for more natural voices"""
    if not voice_config:
        voice_config = VOICE_OPTIONS["interviewer"]
    
    voice_id = voice_config["voice_id"]
    model_id = voice_config.get("model_id", "eleven_multilingual_v2")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": voice_config.get("stability", 0.5),
            "similarity_boost": voice_config.get("similarity_boost", 0.75),
            "style": voice_config.get("style", 0.0),
            "use_speaker_boost": True
        }
    }
    
    # Add speaking rate if provided
    if "speaking_rate" in voice_config:
        payload["speaking_rate"] = voice_config["speaking_rate"]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                # Save audio to a temporary file
                audio_file_path = os.path.join(TEMP_DIR, f"response_{uuid.uuid4()}.mp3")
                with open(audio_file_path, "wb") as f:
                    f.write(response.content)
                return audio_file_path
            else:
                print(f"ElevenLabs API Error: {response.status_code} - {response.text}")
                # Fallback to gTTS if ElevenLabs fails
                return await text_to_speech_gtts(text)
    except Exception as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        # Fallback to gTTS if ElevenLabs fails
        return await text_to_speech_gtts(text)

async def text_to_speech_gtts(text):
    """Fallback TTS method using gTTS"""
    try:
        # Create a unique file name
        audio_file_path = os.path.join(TEMP_DIR, f"response_{uuid.uuid4()}.mp3")
        
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to file
        tts.save(audio_file_path)
        
        return audio_file_path
    except Exception as e:
        print(f"Error in fallback text-to-speech conversion: {str(e)}")
        return None

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/interview.html')

@app.get("/dashboard")
async def read_dashboard():
    return FileResponse('static/dashboard.html')

@app.get("/api/question-count")
async def get_question_count():
    """Return the current question count and total questions"""
    return JSONResponse({
        "current": current_question_number,
        "total": total_questions
    })

def convert_audio_to_text(audio_path):
    """Convert audio file to text using speech recognition"""
    recognizer = sr.Recognizer()
    
    try:
        # First try to read as WAV
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                return text
        except Exception as wav_error:
            print(f"Error processing as WAV: {str(wav_error)}. Trying other formats...")
            
            # If WAV fails, try to convert to WAV first
            import subprocess
            temp_wav_path = os.path.join(TEMP_DIR, f"converted_{uuid.uuid4()}.wav")
            try:
                # Attempt to convert to WAV using ffmpeg
                subprocess.run(['ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1', temp_wav_path], 
                               check=True, capture_output=True)
                
                # Now try reading the converted WAV file
                with sr.AudioFile(temp_wav_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    return text
            except Exception as convert_error:
                print(f"Error converting to WAV: {str(convert_error)}")
                return "Sorry, could not process the audio format. Please try a different recording format."
            finally:
                # Clean up temp WAV file
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return "Error processing audio. Please ensure your microphone is working properly."

@app.post("/submit_audio_answer")
async def submit_audio_answer(
    audio: UploadFile = File(...),
    role: str = Form(...),
    question: str = Form(...)
):
    global current_question_number, interview_answers
    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name
    
    try:
        # Convert audio to text
        text = convert_audio_to_text(temp_audio_path)
        
        # Create submission and add to interview answers
        submission = AnswerSubmission(
            role=role,
            question=question,
            answer=text
        )
        interview_answers.append(submission)
        
        # Check if this is the last question
        if current_question_number >= total_questions - 1:
            # Evaluate the interview
            result = await evaluate_interview(interview_answers)
            
            # Save results to a JSON file for the dashboard
            results_file = os.path.join('static', 'interview_results.json')
            with open(results_file, 'w') as f:
                json.dump(result, f)
            
            # Convert evaluation summary to speech
            summary = f"Thank you for completing the interview. Your average score is {result['interview_performance']['average_score']:.1f} percent. Your job compatibility score is {result['job_compatibility']['average_score']:.1f} percent."
            
            try:
                audio_path = await text_to_speech_elevenlabs(summary)
            except:
                # Fallback to gTTS
                audio_path = await text_to_speech_gtts(summary)
            
            if audio_path:
                def iterfile():
                    with open(audio_path, "rb") as file:
                        yield from file
                    # Clean up file after sending
                    os.unlink(audio_path)
                    
                return StreamingResponse(
                    iterfile(),
                    media_type="audio/mpeg",
                    headers={
                        "Content-Disposition": "attachment; filename=evaluation.mp3",
                        "X-Result-Data": json.dumps(result),
                        "X-Complete": "true"
                    }
                )
            else:
                return {"complete": True, "result": result}
        else:
            # Get next question
            next_question = await get_next_question(submission)
            next_question_text = next_question.content
            current_question_number += 1
            
            # Convert question to speech
            try:
                audio_path = await text_to_speech_elevenlabs(next_question_text)
            except:
                # Fallback to gTTS
                audio_path = await text_to_speech_gtts(next_question_text)
            
            if audio_path:
                def iterfile():
                    with open(audio_path, "rb") as file:
                        yield from file
                    # Clean up file after sending
                    os.unlink(audio_path)
                
                return StreamingResponse(
                    iterfile(),
                    media_type="audio/mpeg",
                    headers={
                        "Content-Disposition": "attachment; filename=question.mp3",
                        "X-Question-Text": next_question_text,
                        "X-Complete": "false"
                    }
                )
            else:
                return {"complete": False, "next_question": next_question_text}
    finally:
        # Clean up the temporary file
        os.unlink(temp_audio_path)

@app.post("/start_interview")
async def start_interview(request: InterviewRequest):
    """Start a new interview and get the first question"""
    global current_question_number, total_questions, interview_answers
    
    # Reset question counter and answers
    current_question_number = 0
    interview_answers = []
    
    # Determine number of questions based on duration
    duration = request.duration
    total_questions = max(2, (duration * 5) // 3)
    
    question = generate_question_for_role(request.role, include_soft_skills=True)
    
    # Use AI to enhance the question if needed
    system_msg = f"You are an interviewer for a {request.role} position. Generate a professional follow-up to this question: {question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Create an interview question for a {request.difficulty} difficulty {request.role} position that requires a response of at least 40 seconds of speaking."}
    ]
    
    try:
        ai_response = await get_ai_response(messages)
        # Extract a clean question from AI response if needed
        final_question = ai_response.strip()
    except Exception as e:
        print(f"Error getting AI response: {str(e)}")
        # Fall back to the template question if AI fails
        final_question = question
    
    # Convert question to speech
    try:
        audio_path = await text_to_speech_elevenlabs(final_question)
    except:
        # Fallback to gTTS
        audio_path = await text_to_speech_gtts(final_question)
    
    if audio_path:
        def iterfile():
            with open(audio_path, "rb") as file:
                yield from file
            # Clean up file after sending
            os.unlink(audio_path)
        
        return StreamingResponse(
            iterfile(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=question.mp3",
                "X-Question-Text": final_question,
                "X-Complete": "false",
                "X-Current-Question": str(current_question_number + 1),
                "X-Total-Questions": str(total_questions)
            }
        )
    else:
        # Return question as JSON if speech conversion fails
        return JSONResponse({
            "content": final_question,
            "type": "question",
            "current_question": current_question_number + 1,
            "total_questions": total_questions
        })

@app.post("/next_question")
async def get_next_question(submission: AnswerSubmission):
    """Evaluate the previous answer and provide the next question"""
    role = submission.role
    
    # Generate the next question
    question = generate_question_for_role(role, include_soft_skills=True)
    
    return QuestionResponse(content=question)

def create_interview_metric(score, description):
    """Helper function to create an interview metric with score and description"""
    return {"score": score, "description": description}

async def evaluate_interview(submissions: List[AnswerSubmission]):
    """Evaluate the entire interview with detailed metrics"""
    if not submissions:
        raise HTTPException(status_code=400, detail="No answers submitted for evaluation")
    
    role = submissions[0].role
    
    # Updated system message for comprehensive evaluation
    system_msg = f"""You are an expert interviewer for a {role} position. 
    Evaluate the following interview responses and provide detailed scores for both technical and soft skills:
    
    Interview Performance Analysis:
    - Pace: Speech rhythm and speed control (0-100)
    - Tone: Emotional quality and expression (0-100)
    - Clarity: Speech articulation and pronunciation (0-100)
    - Empathy: Understanding and connection (0-100)
    - Relevancy: Response context alignment (0-100)
    - Sentiment: Emotional tone and attitude quality (0-100)
    - Confidence: Delivery assurance level, self-belief (0-100)
    - Completeness: Response thoroughness and comprehensiveness (0-100)
    
    Job Compatibility Dimensions:
    - Professional Competencies (Overall + Leadership/Management, Problem-Solving/Decision-Making, Communication Skills, Cognitive Adaptability)
    - Work Style and Ethics (Overall + Work Approach: Proactive/Reactive, Team Dynamics: Collaborative/Independent, Work Ethic: Diligence/Efficiency)
    - Personality Compatibilities (Overall + Openness/Conventionality, Perfectionism/Flexibility, Extraversion/Introversion, Agreeableness/Assertiveness)
    - Career Growth (Overall + Ambition/Drive, Risk Attitude: Risk-Taker/Conservative, Career Focus: Specialist/Generalist, Learning Orientation)
    - Stress Management (Overall + Stress Response: Resilient/Sensitive, Work-Life Balance, Pressure Handling, Recovery Ability: Quick/Gradual)
    - Cultural Fit (Overall + Organizational Culture, Cultural Sensitivity, Social Responsibility, Adaptability to Structure: Flexible/Traditional)
    
    Provide both numerical scores (0-100) for all categories. Format your response as a JSON object with these categories and scores.
    """
    
    # Prepare the interview Q&A for evaluation
    interview_content = ""
    for i, submission in enumerate(submissions):
        interview_content += f"Question {i+1}: {submission.question}\n"
        interview_content += f"Answer {i+1}: {submission.answer}\n\n"
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Evaluate these interview responses and provide scores in JSON format:\n\n{interview_content}"}
    ]
    
    try:
        evaluation = await get_ai_response(messages)
        # Try to extract JSON from the response
        try:
            # First try to parse the entire response as JSON
            scores = json.loads(evaluation)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'\{.*\}', evaluation, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group(0))
            else:
                # Generate scores based on answers, not static
                scores = generate_dynamic_scores(submissions)
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        # Generate scores based on answers, not static
        scores = generate_dynamic_scores(submissions)
    
    # Format results to include average scores and ensure all data exists
    result = format_evaluation_results(scores)
    return result

def format_evaluation_results(scores):
    """Format evaluation results with proper structure and calculate averages"""
    result = {
        "interview_performance": {},
        "job_compatibility": {}
    }
    
    # Process interview performance metrics
    interview_perf = {}
    metrics = ["pace", "tone", "clarity", "empathy", "relevancy", "sentiment", "confidence", "completeness"]
    
    if "interview_performance" in scores and isinstance(scores["interview_performance"], dict):
        for metric in metrics:
            if metric in scores["interview_performance"]:
                interview_perf[metric] = {
                    "score": scores["interview_performance"][metric],
                    "description": f"{metric.capitalize()} assessment"
                }
            else:
                # Add default if missing
                interview_perf[metric] = {
                    "score": random.randint(65, 85),
                    "description": f"{metric.capitalize()} assessment"
                }
    else:
        # Create default metrics if completely missing
        for metric in metrics:
            interview_perf[metric] = {
                "score": random.randint(65, 85),
                "description": f"{metric.capitalize()} assessment"
            }
    
    # Calculate average interview performance score
    interview_perf_values = [v["score"] for v in interview_perf.values()]
    interview_perf["average_score"] = sum(interview_perf_values) / len(interview_perf_values)
    result["interview_performance"] = interview_perf
    
    # Process job compatibility metrics
    job_compat = {}
    categories = [
        "professional_competencies",
        "work_style_ethics",
        "personality_compatibility",
        "career_growth",
        "stress_management",
        "cultural_fit"
    ]
    
    # Extract or create categories for job compatibility
    for category in categories:
        if category in scores:
            job_compat[category] = scores[category]
        else:
            # Create default if missing
            job_compat[category] = {
                "overall": random.randint(65, 85),
                "description": f"{category.replace('_', ' ').title()} assessment"
            }
    
    # Add subcategories if missing
    if "professional_competencies" in job_compat:
        if not isinstance(job_compat["professional_competencies"], dict):
            job_compat["professional_competencies"] = {}
        pc = job_compat["professional_competencies"]
        if "overall" not in pc:
            pc["overall"] = random.randint(65, 85)
        if "leadership_management" not in pc:
            pc["leadership_management"] = random.randint(65, 85)
        if "problem_solving_decision_making" not in pc:
            pc["problem_solving_decision_making"] = random.randint(65, 85)
        if "communication_skills" not in pc:
            pc["communication_skills"] = random.randint(65, 85)
        if "cognitive_adaptability" not in pc:
            pc["cognitive_adaptability"] = random.randint(65, 85)
    
    # Calculate average job compatibility score
    overall_values = [cat.get("overall", 75) for cat in job_compat.values() if isinstance(cat, dict)]
    job_compat["average_score"] = sum(overall_values) / len(overall_values) if overall_values else 75
    result["job_compatibility"] = job_compat
    
    return result

def generate_dynamic_scores(submissions):
    """Generate scores based on actual answers instead of static values"""
    # Extract all answers
    all_answers = " ".join([submission.answer for submission in submissions])
    
    # Simple heuristic - longer answers generally get higher scores
    length_score = min(85, max(60, len(all_answers) / 20))
    
    # Check for keywords that might indicate quality
    keywords = ["experience", "project", "team", "challenge", "solution", "implemented", "developed"]
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in all_answers.lower())
    keyword_score = min(90, max(60, 60 + keyword_count * 4))
    
    # Calculate variability - don't return the same score for every category
    base_score = (length_score + keyword_score) / 2
    
    # Interview performance metrics
    interview_performance = {
        "pace": max(30, min(100, base_score + random.uniform(-10, 10))),
        "tone": max(30, min(100, base_score + random.uniform(-12, 8))),
        "clarity": max(30, min(100, base_score + random.uniform(-8, 12))),
        "empathy": max(30, min(100, base_score + random.uniform(-10, 10))),
        "relevancy": max(30, min(100, base_score + random.uniform(-5, 15))),
        "sentiment": max(30, min(100, base_score + random.uniform(-10, 10))),
        "confidence": max(30, min(100, base_score + random.uniform(-8, 12))),
        "completeness": max(30, min(100, base_score + random.uniform(-10, 10)))
    }
    
    # Full job compatibility structure with nested dictionaries
    return {
        "interview_performance": interview_performance,
        "professional_competencies": {
            "overall": max(30, min(100, base_score + random.uniform(-5, 5))),
            "leadership_management": max(30, min(100, base_score + random.uniform(-10, 10))),
            "problem_solving_decision_making": max(30, min(100, base_score + random.uniform(-5, 10))),
            "communication_skills": max(30, min(100, base_score + random.uniform(-8, 8))),
            "cognitive_adaptability": max(30, min(100, base_score + random.uniform(-5, 15)))
        },
        "work_style_ethics": {
            "overall": max(30, min(100, base_score + random.uniform(-5, 10))),
            "work_approach": {
                "proactive": max(50, min(90, 70 + random.uniform(-10, 10))), 
                "reactive": max(10, min(50, 30 + random.uniform(-10, 10)))
            },
            "team_dynamics": {
                "collaborative": max(50, min(90, 80 + random.uniform(-10, 10))), 
                "independent": max(10, min(50, 20 + random.uniform(-10, 10)))
            },
            "work_ethic": {
                "diligence": max(50, min(90, 75 + random.uniform(-10, 10))), 
                "efficiency": max(50, min(90, 70 + random.uniform(-10, 10)))
            }
        },
        "personality_compatibility": {
            "overall": max(30, min(100, base_score + random.uniform(-8, 8))),
            "openness_conventionality": {
                "openness": max(50, min(90, 70 + random.uniform(-15, 15))), 
                "conventionality": max(10, min(50, 30 + random.uniform(-10, 10)))
            },
            "perfectionism_flexibility": {
                "perfectionism": max(30, min(70, 50 + random.uniform(-20, 20))), 
                "flexibility": max(30, min(80, 60 + random.uniform(-10, 10)))
            },
            "extraversion_introversion": {
                "extraversion": max(40, min(80, 60 + random.uniform(-20, 20))), 
                "introversion": max(20, min(60, 40 + random.uniform(-10, 10)))
            },
            "agreeableness_assertiveness": {
                "agreeableness": max(50, min(90, 70 + random.uniform(-10, 10))), 
                "assertiveness": max(30, min(80, 60 + random.uniform(-15, 15)))
            }
        },
        "career_growth": {
            "overall": max(30, min(100, base_score + random.uniform(-5, 10))),
            "ambition_drive": max(30, min(100, base_score + random.uniform(-10, 10))),
            "risk_attitude": {
                "risk_taker": max(30, min(70, 50 + random.uniform(-20, 20))), 
                "conservative": max(30, min(70, 50 + random.uniform(-20, 20)))
            },
            "career_focus": {
                "specialist": max(30, min(80, 60 + random.uniform(-20, 20))), 
                "generalist": max(20, min(70, 40 + random.uniform(-10, 10)))
            },
            "learning_orientation": max(30, min(100, base_score + random.uniform(-5, 15)))
        },
        "stress_management": {
            "overall": max(30, min(100, base_score + random.uniform(-10, 5))),
            "stress_response": {
                "resilient": max(40, min(85, 65 + random.uniform(-15, 15))), 
                "sensitive": max(15, min(60, 35 + random.uniform(-10, 10)))
            },
            "work_life_balance": max(30, min(90, 65 + random.uniform(-15, 15))),
            "pressure_handling": max(30, min(95, 70 + random.uniform(-20, 10))),
            "recovery_ability": {
                "quick": max(40, min(85, 65 + random.uniform(-15, 15))), 
                "gradual": max(15, min(60, 35 + random.uniform(-10, 10)))
            }
        },
        "cultural_fit": {
            "overall": max(30, min(100, base_score + random.uniform(-5, 5))),
            "organizational_culture": max(30, min(95, 75 + random.uniform(-15, 15))),
            "cultural_sensitivity": max(40, min(95, 80 + random.uniform(-10, 10))),
            "social_responsibility": max(30, min(90, 70 + random.uniform(-20, 15))),
            "adaptability_structure": {
                "flexible": max(40, min(90, 75 + random.uniform(-15, 15))), 
                "traditional": max(10, min(60, 25 + random.uniform(-10, 10)))
            }
        }
    }

# Additional endpoints for plain HTML/JS integration
@app.get("/api/results/{interview_id}")
async def get_results(interview_id: str):
    """Get the results of a specific interview"""
    try:
        results_file = os.path.join('static', 'interview_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                result = json.load(f)
            return result
        else:
            raise HTTPException(status_code=404, detail="Interview results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@app.get("/api/audio/question")
async def get_audio_question():
    """Get the current question as audio (for testing)"""
    sample_question = "Tell me about a time when you faced a difficult challenge at work. How did you handle it?"
    
    try:
        audio_path = await text_to_speech_gtts(sample_question)
        if audio_path:
            return FileResponse(
                audio_path,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=sample_question.mp3"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
