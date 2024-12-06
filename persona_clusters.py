#!/usr/bin/env python3

import json
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import re

class PersonaClusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = defaultdict(list)
        
    def extract_features(self, persona):
        """Extract key features from persona text using regex"""
        # Ensure we have a clean string
        text = str(persona).lower().strip()
        
        # Business Role Categories with expanded keywords
        roles = {
            'it_admin': [
                'it administrator', 'system admin', 'network admin', 'it manager', 
                'technology manager', 'cio', 'it director', 'infrastructure', 
                'systems engineer', 'tech lead', 'information technology',
                'software engineer', 'devops', 'system architect'
            ],
            'facilities': [
                'facility manager', 'building manager', 'maintenance manager',
                'operations manager', 'property maintenance', 'facility operations',
                'building maintenance', 'facilities director', 'site manager'
            ],
            'executive': [
                'ceo', 'cto', 'cfo', 'executive', 'director', 'vp', 'chief',
                'president', 'head of', 'founder', 'owner', 'board member',
                'managing director', 'senior executive'
            ],
            'retail': [
                'retail manager', 'store manager', 'merchandising', 
                'retail operations', 'shop owner', 'retail director',
                'sales manager', 'store owner', 'retail supervisor'
            ],
            'multifamily': [
                'property manager', 'leasing manager', 'real estate manager',
                'housing manager', 'apartment manager', 'residential manager',
                'community manager', 'building supervisor', 'property supervisor'
            ]
        }
        
        # Extract role using more flexible matching
        role = None
        for role_type, keywords in roles.items():
            if any(keyword in text for keyword in keywords):
                role = role_type
                break
        
        return {
            "text": text,
            "original": persona,  # Keep original text
            "role": role
        }
    
    def cluster_personas(self, personas):
        """Cluster personas based on business roles"""
        print("Analyzing personas...")
        
        # Group personas by role
        role_clusters = defaultdict(list)
        for persona in personas:
            if persona["role"]:
                # Convert role to uppercase for consistency
                role_key = "IT_ADMIN" if persona["role"] == "it_admin" else persona["role"].upper()
                # Store original text in the cluster
                original_text = str(persona["original"]).strip()
                if original_text:  # Only add non-empty personas
                    role_clusters[role_key].append(original_text)
        
        # Analyze each role cluster
        print("\nRole-based Analysis:")
        for role, cluster in role_clusters.items():
            print(f"\n=== {role} Cluster ({len(cluster)} personas) ===")
            
            # Show sample personas
            if cluster:
                print("\nSample Personas:")
                for p in cluster[:3]:
                    print(f"- {p}")
            else:
                print("\nNo personas found in this cluster.")
        
        return role_clusters
    
    def get_cluster_personas(self, cluster_id, n_samples=10):
        """Get n_samples personas from a specific cluster"""
        if cluster_id in self.clusters:
            personas = self.clusters[cluster_id]
            if n_samples > len(personas):
                return personas
            return np.random.choice(personas, n_samples, replace=False)
        return []

def main():
    # Load personas
    print("Loading personas...")
    dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")
    personas = dataset['train']['persona']
    
    # Print dataset info
    print("\n=== Dataset Info ===")
    print(f"Total entries: {len(personas)}")
    print("Type: String entries (one persona description per line)")
    
    # Show data distribution
    print("\n=== Content Analysis ===")
    word_counts = [len(str(p).split()) for p in personas[:1000]]
    avg_words = sum(word_counts) / len(word_counts)
    print(f"Average words per persona: {avg_words:.1f}")
    
    # Show sample personas
    print("\n=== Sample Personas ===")
    for i, persona in enumerate(personas[:10]):
        print(f"\n{i+1}. {persona}")
        print("-" * 50)
    
    proceed = input("\nWould you like to proceed with clustering? (y/n): ")
    if proceed.lower() != 'y':
        return
    
    # Process personas
    print("\nProcessing personas...")
    clusterer = PersonaClusterer(n_clusters=5)
    processed_personas = []
    
    for persona in tqdm(personas[:1000]):  # Start with a subset for testing
        features = clusterer.extract_features(persona)
        processed_personas.append(features)
    
    # Cluster personas
    clusters = clusterer.cluster_personas(processed_personas)
    
    # Save clusters
    with open('persona_clusters.json', 'w') as f:
        json.dump({str(k): [p["text"] for p in v] for k, v in clusters.items()}, f, indent=2)
    
    print("\nClusters saved to persona_clusters.json")

if __name__ == "__main__":
    main() 