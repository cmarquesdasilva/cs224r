"""
Prompt Builder for Personality Generation

This module provides functionality for building structured prompts used in personality generation,
based on personality trait data from the Five-Factor Model (also known as the Big Five or OCEAN model).
It converts raw personality data into formatted prompts using mustache templates.

The module handles:
- Loading and processing personality data from pandas DataFrames
- Converting numeric personality scores into descriptive levels
- Mapping demographic information (age, gender, country)
- Rendering structured templates with mustache syntax

The personality schema includes the five main traits (openness, conscientiousness, extraversion, 
agreeableness, neuroticism) and their 30 facets, aligning with standard psychological assessment models.
"""

import pandas as pd
import chevron
from typing import Dict, List
from dataclasses import dataclass, field
import json

@dataclass
class PersonalitySchema:
    """
    Schema definition for the Five-Factor Model (Big Five) personality structure.
    
    This dataclass defines the hierarchical structure of personality according to the
    Big Five model, with five major traits and 30 facets (6 per trait). This structure
    is used to organize and interpret personality assessment data.
    
    Attributes:
        traits: List of the five major personality traits (OCEAN)
        facets: Dictionary mapping each trait to its six constituent facets
    """

    traits: List[str] = field(default_factory=lambda: [
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
    ])
    facets: Dict[str, List[str]] = field(default_factory=lambda: {
        "openness": [
            "imagination", "artistic_interests", "emotionality",
            "adventurousness", "intellect", "liberalism"
        ],
        "conscientiousness": [
            "self_efficacy", "orderliness", "dutifulness",
            "achievement_striving", "self_discipline", "cautiousness"
        ],
        "extraversion": [
            "friendliness", "gregariousness", "assertiveness",
            "active_level", "excitement_seeking", "cheerfulness"
        ],
        "agreeableness": [
            "trust", "morality", "altruism",
            "cooperation", "modesty", "sympathy"
        ],
        "neuroticism": [
            "anxiety", "anger", "depression",
            "self_consciousness", "immoderation", "vulnerability"
        ]
    })

class PromptBuilder:
    """
    Builds structured prompts for personality generation based on personality assessment data.
    
    This class converts raw personality trait and facet data from a DataFrame into 
    formatted prompts using mustache templates. It handles the extraction and preparation
    of personality profile data, demographic information, and integration with prompt templates.
    
    Attributes:
        data: DataFrame containing personality assessment data and demographics
        template_path: Path to the mustache template file used for rendering
        schema: PersonalitySchema defining the structure of personality traits and facets
    """

    def __init__(self, data: pd.DataFrame, template_path: str):
        """
        Initialize the PromptBuilder with data and template.
        
        Args:
            data: DataFrame containing personality scores and demographic information
            template_path: Path to the mustache template file for prompt generation
        """
        self.data = data
        self.template_path = template_path
        self.schema = PersonalitySchema()

    def _get_person_data(self, index: int) -> Dict:
        """
        Extract personality profile and demographic data for a specific individual.
        
        This method processes a row from the input DataFrame, extracting trait and facet
        levels along with demographic information to create a context dictionary for
        template rendering.
        
        Args:
            index: Row index in the DataFrame to extract data from
            
        Returns:
            Dictionary containing personality profile (traits and facets) and 
            demographic information (age, sex, country)
            
        Note:
            The method converts numeric sex codes to labels (1=Male, 2=Female)
            and handles missing facet data by labeling it as "Unknown"
        """
        row = self.data.iloc[index]
        # Convert numeric sex to label
        sex = "Female" if row["sex"] == 2 else "Male"

        # Extract trait levels
        traits = {
            trait: row[f"{trait}_level"]
            for trait in self.schema.traits
        }

        # Extract facet levels
        facets = {}
        for trait, facet_list in self.schema.facets.items():
            for facet in facet_list:
                matches = [col for col in self.data.columns if col.startswith(f"{facet}_level")]
                if matches:
                    facets[facet] = row[matches[0]]
                else:
                    facets[facet] = "Unknown"

        return {
            "personality_profile": {
                **traits,
                **facets
            },
            "age": row["age"],
            "sex": sex,
            "country": row["country"]
        }

    def build_prompt(self, index: int) -> str:
        """
        Generate a complete personality prompt for the specified data index.
        
        This method builds a complete prompt by extracting the relevant personality
        and demographic data, then rendering it with the specified mustache template.
        
        Args:
            index: Row index in the DataFrame to build the prompt for
            
        Returns:
            Rendered prompt string containing personality specifications
            
        Raises:
            IndexError: If the index is out of bounds for the DataFrame
            FileNotFoundError: If the template file doesn't exist
        """
        context = self._get_person_data(index)
        with open(self.template_path, 'r') as f:
            template = f.read()
        return chevron.render(template, context)
