# House of 200,000 Synthetic Personae

A powerful toolkit for working with synthetic personas, built on top of the PERSONA HUB dataset. This project provides tools for persona analysis, clustering, and message testing across different business roles.

## Features

- **Persona Synthesis** (`main.py`): Generate specialized content from personas including:
  - Math problems tailored to persona backgrounds
  - Instructions based on persona expertise
  - NPC characters for gaming contexts
  - Knowledge articles from persona perspectives

- **Persona Clustering** (`persona_clusters.py`): 
  - Automated clustering of personas by business roles
  - Support for IT Admin, Executive, Facilities, Retail, and Multifamily roles
  - Feature extraction based on role-specific keywords
  - Demographic and expertise-based grouping

- **Message Testing** (`message_tester.py`):
  - Test marketing messages against specific persona clusters
  - Get detailed resonance scores and feedback
  - Analyze message effectiveness across different roles
  - Generate improvement suggestions based on persona responses

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

3. Run persona synthesis:
```bash
python main.py
```

4. Test messages against personas:
```bash
python message_tester.py --messages_file messages.json --output_file results.json --role it_admin
```

## Usage Examples

### Persona Synthesis
```python
python main.py --template math --sample_size 10
python main.py --template npc --sample_size 10
python main.py --template knowledge --sample_size 10
```

### Message Testing
```python
# Test IT-focused messages
python message_tester.py --messages_file it_messages.json --output_file it_results.json --role it_admin

# Test executive-focused messages
python message_tester.py --messages_file exec_messages.json --output_file exec_results.json --role executive
```

### Persona Clustering
```python
# Cluster personas and analyze demographics
python persona_clusters.py
```

## Project Structure

- `main.py` - Main script for persona synthesis
- `message_tester.py` - Tool for testing messages against persona clusters
- `persona_clusters.py` - Persona clustering and analysis
- `code/prompt_templates.py` - Templates for different synthesis tasks
- `requirements.txt` - Project dependencies

## Output Formats

### Message Testing Results
```json
{
    "message": "Your message here",
    "average_score": 7.5,
    "detailed_responses": [
        {
            "resonance_score": 8,
            "technical_accuracy": "...",
            "operational_impact": "...",
            "security_considerations": "...",
            "implementation_concerns": [...]
        }
    ],
    "key_themes": {...}
}
```

### Persona Clusters
```json
{
    "IT_ADMIN": [...],
    "EXECUTIVE": [...],
    "FACILITIES": [...],
    "RETAIL": [...],
    "MULTIFAMILY": [...]
}
```

## Disclaimer

This toolkit facilitates synthetic data creation at scale to simulate diverse inputs from a wide variety of personas. While powerful, it comes with important ethical considerations. Please review the [full disclaimer and ethical considerations](https://github.com/tencent-ailab/persona-hub?tab=readme-ov-file#disclaimer) from the original PERSONA HUB project.

## Credits

This project is built on top of the [PERSONA HUB](https://huggingface.co/datasets/proj-persona/PersonaHub) dataset and toolkit ([GitHub](https://github.com/tencent-ailab/persona-hub)), a comprehensive collection of 200,000 synthetic personas created by Tencent AI Lab. We acknowledge and thank the creators of PERSONA HUB for providing this valuable resource.
