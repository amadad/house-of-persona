#!/usr/bin/env python3

import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from code.prompt_templates import (it_admin_template, facilities_template, 
                                 executive_template, retail_template, multifamily_template)
from datasets import load_dataset
from persona_clusters import PersonaClusterer
from pydantic import BaseModel
from typing import List

class MessageResponse(BaseModel):
    resonance_score: int
    technical_accuracy: str
    operational_impact: str
    security_considerations: str
    implementation_concerns: List[str]

load_dotenv()
client = OpenAI()

def get_response(prompt: str) -> MessageResponse:
    """Get structured response from OpenAI API"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": """You are an expert in B2B marketing and audience analysis.
                Evaluate the message's resonance with the given persona.
                IMPORTANT: resonance_score must be an integer between 1 and 10."""},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        # Parse response
        response_text = completion.choices[0].message.content
        response_data = json.loads(response_text)
        
        # Validate score range
        score = response_data.get("resonance_score")
        if not isinstance(score, int) or score < 1 or score > 10:
            print(f"\nInvalid score: {score}")
            return None
            
        return MessageResponse(**response_data)
        
    except Exception as e:
        print(f"\nAPI error: {str(e)}")
        return None

def test_messages(messages, role_clusters, template_type, num_personas=10):
    """Test messages against personas of a specific role"""
    templates = {
        "it_admin": it_admin_template,
        "facilities": facilities_template,
        "executive": executive_template,
        "retail": retail_template,
        "multifamily": multifamily_template
    }
    
    if template_type not in templates:
        raise ValueError(f"Invalid template type. Choose from: {', '.join(templates.keys())}")
    
    template = templates[template_type]
    
    # Convert role name to match clustering format
    cluster_role = "IT_ADMIN" if template_type == "it_admin" else template_type.upper()
    personas = role_clusters.get(cluster_role, [])
    
    if not personas:
        print(f"Available clusters: {list(role_clusters.keys())}")
        raise ValueError(f"No personas found for role: {template_type} (cluster key: {cluster_role})")
    
    # Limit number of personas to test
    test_personas = personas[:num_personas]
    print(f"\nTesting messages against {len(test_personas)} {template_type} personas...")
    
    results = []
    for message in tqdm(messages, desc="Testing messages"):
        message_results = []
        for persona in test_personas:
            try:
                # Format prompt with persona and message
                prompt = template.format(
                    persona=str(persona).strip(),
                    message=str(message).strip()
                )
                
                # Get structured response
                response = get_response(prompt)
                if response:
                    message_results.append(response.model_dump())
                
            except Exception as e:
                print(f"\nError processing persona: {str(e)}")
                continue
        
        if not message_results:
            print(f"\nNo valid responses for message: {message[:50]}...")
            continue
        
        # Calculate average score
        scores = [r["resonance_score"] for r in message_results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        insights = {
            "message": message,
            "average_score": avg_score,
            "detailed_responses": message_results,
            "key_themes": analyze_themes(message_results)
        }
        
        results.append(insights)
    
    # Sort by average score
    results.sort(key=lambda x: x["average_score"], reverse=True)
    return results

def analyze_themes(responses):
    """Extract common themes from responses"""
    themes = {
        "strengths": [],
        "concerns": [],
        "suggestions": []
    }
    
    for response in responses:
        # Extract positive aspects
        for key in response:
            if any(term in key.lower() for term in ['impact', 'benefit', 'advantage', 'efficiency']):
                value = response[key]
                if isinstance(value, str) and len(value) > 5:
                    themes["strengths"].append(value)
        
        # Extract concerns and challenges
        for key in response:
            if any(term in key.lower() for term in ['concern', 'challenge', 'consideration']):
                value = response[key]
                if isinstance(value, list):
                    themes["concerns"].extend(value)
    
    # Deduplicate and get top themes
    themes["strengths"] = list(set(themes["strengths"]))[:3]
    themes["concerns"] = list(set(themes["concerns"]))[:3]
    
    return themes

def main():
    parser = argparse.ArgumentParser(description="Test marketing messages against specific audience segments")
    parser.add_argument("--messages_file", required=True, help="JSON file containing messages to test")
    parser.add_argument("--output_file", required=True, help="Where to save results")
    parser.add_argument("--role", choices=["it_admin", "facilities", "executive", "retail", "multifamily"], required=True)
    parser.add_argument("--num_personas", type=int, default=10, help="Number of personas per role to test against")
    
    args = parser.parse_args()
    
    # Load messages
    with open(args.messages_file) as f:
        data = json.load(f)
        messages = data.get("messages", [])
        if not messages:
            raise ValueError("No messages found in the input file")
    
    # Load and cluster personas
    print("Loading and clustering personas...")
    dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")
    personas = dataset['train']['persona']
    
    clusterer = PersonaClusterer()
    processed_personas = []
    
    for persona in tqdm(personas[:1000], desc="Processing personas"):
        features = clusterer.extract_features(persona)
        processed_personas.append(features)
    
    role_clusters = clusterer.cluster_personas(processed_personas)
    
    # Run tests
    results = test_messages(messages, role_clusters, args.role, args.num_personas)
    
    # Save detailed results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary of top 3 messages
    print("\nTop 3 Most Resonant Messages:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. Score: {result['average_score']:.2f}")
        print(f"Message: {result['message']}")
        print("\nKey Themes:")
        print("Strengths:", result['key_themes']['strengths'])
        print("Concerns:", result['key_themes']['concerns'])

if __name__ == "__main__":
    main() 