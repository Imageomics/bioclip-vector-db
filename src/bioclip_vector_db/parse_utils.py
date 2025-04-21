"""Utility containing all parsing functions."""

import re
import logging

from typing import List, Dict


def parse_taxontag_com(tag: str, logger: logging.Logger) -> Dict:
    """Helper function to parse and process the taxon tags.

    These taxon tags are stored in the column "taxontag_com.txt" in the
    bioclip dataset. The tags are in the format:
    a photo of kingdom <kingdom> phylum <phylum> class <class> order <order>
    family <family> genus <genus> species <species> with common name <common name>.

    The function returns a dictionary with the keys:
    kingdom, phylum, class, order, family, genus, and common name.
    """

    # this regex supports non-ascii characters
    # in the taxa classifications.
    # Matches "a photo of" followed by optional taxonomic ranks (kingdom, phylum, etc.) 
    # and an optional common name, ending with a period.
    regex = re.compile(
        r"a photo of"
        r"(?: kingdom (.*?)(?= phylum| class| order| family| genus| species| with common name|\.))?"
        r"(?: phylum (.*?)(?= class| order| family| genus| species| with common name|\.))?"
        r"(?: class (.*?)(?= order| family| genus| species| with common name|\.))?"
        r"(?: order (.*?)(?= family| genus| species| with common name|\.))?"
        r"(?: family (.*?)(?= genus| species| with common name|\.))?"
        r"(?: genus (.*?)(?= species| with common name|\.))?"
        r"(?: species (.*?)(?= with common name|\.))?"
        r"(?: with common name (.*?))?\."
    )
    rank_keys = [
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "common name",
    ]

    match = regex.search(tag)

    taxa_ranks = {rank: "" for rank in rank_keys}

    if match:
        taxa_ranks = {
            key: val.strip() if val is not None else ""
            for key, val in dict(zip(rank_keys, match.groups())).items()
        }
    else:
        logger.warning(f"Failed to parse taxon tag: {tag}\n")
        
    taxa_ranks["raw_tag"] = tag
    
    

    # return default values.
    # logger.info(f"Parsed taxon tag: {taxa_ranks}\n")
    return taxa_ranks
