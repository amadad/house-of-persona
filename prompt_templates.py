brand_template = """
Given this persona: {persona}
Evaluate this brand message: {message}

Return a JSON with:
- resonance_score (1-10)
- emotional_response
- key_resonating_elements
- points_of_disconnect
"""

retail_template = """
As a retail consumer with this background: {persona}
Evaluate this retail message: {message}

Return a JSON with:
- purchase_intent_score (1-10)
- likelihood_to_visit
- price_sensitivity_reaction
- product_interest_level
"""

multifamily_template = """
As a potential resident with this background: {persona}
Evaluate this property message: {message}

Return a JSON with:
- lifestyle_fit_score (1-10)
- location_appeal
- amenity_importance
- price_value_perception
""" 