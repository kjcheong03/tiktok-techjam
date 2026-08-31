"""Bounded, manifest-driven experiment orchestration."""

from ghostlab.campaign.catalog import TechniqueCatalog, load_catalog
from ghostlab.campaign.models import CampaignManifest, CandidateSpec, TechniqueSpec

__all__ = [
    "CampaignManifest",
    "CandidateSpec",
    "TechniqueCatalog",
    "TechniqueSpec",
    "load_catalog",
]
