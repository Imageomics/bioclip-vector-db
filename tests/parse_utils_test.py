import unittest
from unittest.mock import MagicMock
import src.bioclip_vector_db.parse_utils as parse_utils


class TestParseUtils(unittest.TestCase):
    def test_empty_string_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com("", MagicMock()),
            {
                "kingdom": "",
                "phylum": "",
                "class": "",
                "order": "",
                "family": "",
                "genus": "",
                "species": "",
                "common name": "",
            },
        )

    def test_no_taxa_tags_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
                "a photo of kingdom  phylum  class  order  family  genus  "
                "species  with common name .",
                MagicMock(),
            ),
            {
                "kingdom": "",
                "phylum": "",
                "class": "",
                "order": "",
                "family": "",
                "genus": "",
                "species": "",
                "common name": "",
            },
        )

    def test_all_tags_present_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
                "a photo of kingdom a_kingdom phylum a_phylum class a_class order "
                "an_order family a_family genus a_genus species a_species with common name a_common_name.",
                MagicMock(),
            ),
            {
                "kingdom": "a_kingdom",
                "phylum": "a_phylum",
                "class": "a_class",
                "order": "an_order",
                "family": "a_family",
                "genus": "a_genus",
                "species": "a_species",
                "common name": "a_common_name",
            },
        )

    def test_all_tags_with_space_present_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
                "a photo of kingdom a_kingdom suffix phylum a_phylum class a_class order "
                "an_order family a_family genus a_genus species a_species suffix with common name "
                "a_common_name with long suffix.",
                MagicMock(),
            ),
            {
                "kingdom": "a_kingdom suffix",
                "phylum": "a_phylum",
                "class": "a_class",
                "order": "an_order",
                "family": "a_family",
                "genus": "a_genus",
                "species": "a_species suffix",
                "common name": "a_common_name with long suffix",
            },
        )

    def test_no_common_name_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
                "a photo of kingdom a_kingdom phylum a_phylum class a_class order "
                "an_order family a_family genus a_genus species a_species with common name .",
                MagicMock(),
            ),
            {
                "kingdom": "a_kingdom",
                "phylum": "a_phylum",
                "class": "a_class",
                "order": "an_order",
                "family": "a_family",
                "genus": "a_genus",
                "species": "a_species",
                "common name": "",
            },
        )

    def test_missing_species_and_common_name_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
                "a photo of kingdom a_kingdom phylum a_phylum class a_class order "
                "an_order family a_family genus a_genus species  with common name .",
                MagicMock(),
            ),
            {
                "kingdom": "a_kingdom",
                "phylum": "a_phylum",
                "class": "a_class",
                "order": "an_order",
                "family": "a_family",
                "genus": "a_genus",
                "species": "",
                "common name": "",
            },
        )

    def test_missing_kingdom_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
                "a photo of kingdom  phylum a_phylum class a_class order "
                "an_order family a_family genus a_genus species a_species with common name a_common_name.",
                MagicMock(),
            ),
            {
                "kingdom": "",
                "phylum": "a_phylum",
                "class": "a_class",
                "order": "an_order",
                "family": "a_family",
                "genus": "a_genus",
                "species": "a_species",
                "common name": "a_common_name",
            },
        )

    def test_only_species_common_name_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
               "a photo of species species with common name common_name.",
                MagicMock(),
            ),
            {
                "kingdom": "",
                "phylum": "",
                "class": "",
                "order": "",
                "family": "",
                "genus": "",
                "species": "species",
                "common name": "common_name",
            },
        )

    def test_only_kingdom_species_ok(self):
        self.assertDictEqual(
            parse_utils.parse_taxontag_com(
               "a photo of kingdom a_kingdom species a_species.",
                MagicMock(),
            ),
            {
                "kingdom": "a_kingdom",
                "phylum": "",
                "class": "",
                "order": "",
                "family": "",
                "genus": "",
                "species": "a_species",
                "common name": "",
            },
        )
