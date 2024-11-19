import unittest
from pathlib import Path

import tempfile
import json
from utils import (
    create_label_mapping,
    join_text_cols,
    get_labels
)
from data_preprocessing import join_labels, normalize_url

class TestGetLabels(unittest.TestCase):
    def test_get_labels(self):
        data = {
            "category": ["category1", "category2", "category3", "category2", "category1", "category4"]
        }
        expected_labels = ["category1", "category2", "category3", "category4"]
        result = get_labels(data)
        self.assertEqual(result, expected_labels)

    def test_one_category(self):
        data = {"category": ["category1", "category1", "category1"]}
        expected_labels = ["category1"]
        result = get_labels(data)
        self.assertEqual(result, expected_labels)



class TestJoinTextCols(unittest.TestCase):
    def test_join_text_cols(self):
        #row = {"headline": "Breaking News", "short_description": "Fire in the city", "link": "www.example.com"}
        row = {"category": "WEDDINGS", "headline": "The Pre-Wedding Advice You Need to Hear", "authors": "Diane Farr, Contributor\nActress and author", "link": "wedding-planning_us", "short_description": "No one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. And they can really put a damper on things.", "date": "2012-10-02"}
        text_cols = ["headline", "short_description", "link"]
        to_lower = False
        expected_result = {"category": "WEDDINGS", "headline": "The Pre-Wedding Advice You Need to Hear", "authors": "Diane Farr, Contributor\nActress and author", "link": "wedding-planning_us", "short_description": "No one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. And they can really put a damper on things.", "date": "2012-10-02", "input_text": "The Pre-Wedding Advice You Need to Hear No one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. And they can really put a damper on things. wedding-planning_us"}
        result = join_text_cols(row, text_cols, to_lower)
        self.assertEqual(result, expected_result)

    def test_lowercasing(self):
        row = {"category": "WEDDINGS", "headline": "The Pre-Wedding Advice You Need to Hear", "authors": "Diane Farr, Contributor\nActress and author", "link": "wedding-planning_us", "short_description": "No one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. And they can really put a damper on things.", "date": "2012-10-02"}
        text_cols = ["headline", "short_description", "link"]
        to_lower = True
        expected_result = {"category": "WEDDINGS", "headline": "The Pre-Wedding Advice You Need to Hear", "authors": "Diane Farr, Contributor\nActress and author", "link": "wedding-planning_us", "short_description": "No one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. And they can really put a damper on things.", "date": "2012-10-02", "input_text": "the pre-wedding advice you need to hear no one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. and they can really put a damper on things. wedding-planning_us"}
        result = join_text_cols(row, text_cols, to_lower)
        self.assertEqual(result, expected_result)

    def test_single_column_join(self):
        row = {"category": "WEDDINGS", "headline": "The Pre-Wedding Advice You Need to Hear", "authors": "Diane Farr, Contributor\nActress and author", "link": "wedding-planning_us", "short_description": "No one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. And they can really put a damper on things.", "date": "2012-10-02"}
        text_cols = ["headline"]
        to_lower = False
        expected_result = {"category": "WEDDINGS", "headline": "The Pre-Wedding Advice You Need to Hear", "authors": "Diane Farr, Contributor\nActress and author", "link": "wedding-planning_us", "short_description": "No one really warns you that after that magical minute-and-a-half where you and your partner decide to make a lifelong, legally-binding, love-based commitment, a whole mess of other people get involved. And they can really put a damper on things.", "date": "2012-10-02", "input_text": "The Pre-Wedding Advice You Need to Hear"}
        result = join_text_cols(row, text_cols, to_lower)
        self.assertEqual(result, expected_result)

    def test_empty_columns(self):
        row = {"headline": "The Pre-Wedding Advice You Need to Hear", "short_description": "", "link": ""}
        text_cols = ["headline", "short_description", "link"]
        to_lower = False
        expected_result = {"headline": "The Pre-Wedding Advice You Need to Hear", "short_description": "", "link": "", "input_text": "The Pre-Wedding Advice You Need to Hear  "}
        result = join_text_cols(row, text_cols, to_lower)
        self.assertEqual(result, expected_result)


class TestCreateLabelMapping(unittest.TestCase):
    def test_create_label_mapping(self):
        labels = ["Label1", "GLabel", "ALabel", "ZLabel"]
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            out_path = Path(tmpfile.name)
        try:
            result = create_label_mapping(labels, out_path)
            expected_mapping = {"ALabel": 0, "GLabel": 1, "Label1": 2, "ZLabel": 3}

            self.assertEqual(result, expected_mapping)

            with out_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                self.assertEqual(content, expected_mapping)
        finally:
            out_path.unlink()

    def test_single_label(self):
        labels = ["Label1"]
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            out_path = Path(tmpfile.name)
        try:
            result = create_label_mapping(labels, out_path)
            expected_mapping = {"Label1": 0}

            self.assertEqual(result, expected_mapping)

            with out_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                self.assertEqual(content, expected_mapping)
        finally:
            out_path.unlink()


class TestJoinLabels(unittest.TestCase):
    def test_join_labels(self):
        orig_label = 'GREEN'
        cat_dict = {
            "ENVIRONMENT": ["GREEN", "ENVIRONMENT"],
            "PARENTING": ["PARENTING", "PARENTS"],
        }
        expected_result = 'ENVIRONMENT'
        result = join_labels(orig_label, cat_dict)
        self.assertEqual(result, expected_result)

    def test_label_not_in_dict(self):
        orig_label = 'FINANCE'
        cat_dict = {
            "ENVIRONMENT": ["GREEN", "ENVIRONMENT"],
            "PARENTING": ["PARENTING", "PARENTS"],
            "WELLNESS": ["WELLNESS", "HEALTHY LIVING"],
            "BUSINESS": ["BUSINESS", "MONEY"],
        }
        expected_result = 'FINANCE'
        result = join_labels(orig_label, cat_dict)
        self.assertEqual(result, expected_result)

    def test_empty_category_dict(self):
        orig_label = 'PARENTS'
        cat_dict = {}
        expected_result = 'PARENTS'
        result = join_labels(orig_label, cat_dict)
        self.assertEqual(result, expected_result)

    def test_multiple_matches(self):
        # should not happen normaly
        orig_label = 'PARENTS'
        cat_dict = {
            "PARENTING": ["PARENTING", "PARENTS"],
            "FAMILY": ["PARENTS"]
        }
        expected_result = 'PARENTING'
        result = join_labels(orig_label, cat_dict)
        self.assertEqual(result, expected_result)

    def test_case_sensitive_matching(self):
        orig_label = 'parents'
        cat_dict = {
            "PARENTING": ["PARENTING", "PARENTS"]
        }
        expected_result = 'parents'
        result = join_labels(orig_label, cat_dict)
        self.assertEqual(result, expected_result)


class TestNormalizeUrl(unittest.TestCase):
    def test_normalize_url(self):
        url = "https://www.huffingtonpost.com/entry/best-toys-kid-products-prime-day_us_59553139e4b0da2c7321e754"
        expected_result = "best toys kid products prime day us"
        result = normalize_url(url)
        self.assertEqual(result, expected_result)

    def test_url_with_multiple_dashes(self):
        url = "https://www.huffingtonpost.com/entry/my-daughters-mysterious-illness----and-my-own_b_7471504.html"
        expected_result = "my daughters mysterious illness and my own b"
        result = normalize_url(url)
        self.assertEqual(result, expected_result)

    def test_url_with_no_valid_text(self):
        url = "https://example.com/category/_.html"
        expected_result = ""
        result = normalize_url(url)
        self.assertEqual(result, expected_result)

    def test_url_with_only_text(self):
        url = "https://www.huffingtonpost.com/entry/mydaughters.html"
        expected_result = "mydaughters"
        result = normalize_url(url)
        self.assertEqual(result, expected_result)

    def test_empty_url(self):
        url = ""
        expected_result = ""
        result = normalize_url(url)
        self.assertEqual(result, expected_result)

# Run the tests
if __name__ == "__main__":
    unittest.main()
