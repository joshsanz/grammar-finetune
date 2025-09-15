#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test suite for synthetic error insertion functions using unittest.
# Usage: python -m pytest test_synthetic_errors.py -v

import unittest
import sys
import os
sys.path.append(os.path.dirname(__file__))

from insert_synthetic_errors import SyntheticErrorInserter, DEFAULT_CONFIG
import spacy

# Load spaCy model once for all tests
nlp = spacy.load("en_core_web_sm")


class TestSyntheticErrorInsertion(unittest.TestCase):
    """Test suite for synthetic error insertion functionality."""

    def setUp(self):
        """Set up test fixtures with deterministic config."""
        # Create config with 100% error rates for predictable testing
        self.config = {
            'error_rates': {
                'punctuation': {
                    'quote_spaces': 1.0,
                    'missing_quotes': 1.0,
                    'punctuation_outside_quotes': 1.0,
                },
                'typography': {
                    'dropped_letters': 1.0,
                    'swapped_letters': 1.0,
                    'dropped_words': 1.0,
                    'swapped_words': 1.0,
                },
                'spacing': {
                    'spaces_around_punctuation': 1.0,
                    'missing_spaces': 1.0,
                },
                'homophones': {'substitution_rate': 1.0},
                'contractions': {'dropped_nt': 1.0}
            },
            'preserve_ratio': 0.0,  # Never preserve original
            'max_errors_per_chunk': 1,  # Only one error per test
            'min_chunk_length': 1
        }

        # Create inserter with fixed seed for reproducible results
        self.inserter = SyntheticErrorInserter(self.config, seed=12345)

    def test_quote_spacing_error(self):
        """Test that spaces are added inside quotes."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {'quote_spaces': 1.0},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = '"Hello world"'
        result = inserter._insert_punctuation_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        self.assertIn('" Hello world "', modified_text, "Should add spaces inside quotes")

    def test_missing_quotes_error(self):
        """Test that quotes are removed (first, last, or both)."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {'missing_quotes': 1.0},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = '"Hello world"'

        # Test multiple runs to see different removal patterns
        possible_outputs = [
            'Hello world"',  # First quote removed
            '"Hello world',   # Last quote removed
            'Hello world'     # Both quotes removed
        ]

        found_valid_output = False
        for _ in range(10):  # Try multiple times due to randomness
            result = inserter._insert_punctuation_errors(input_text, nlp(input_text))
            modified_text, error_count = result

            if any(expected in modified_text for expected in possible_outputs):
                found_valid_output = True
                break

        self.assertTrue(found_valid_output, f"Should produce one of: {possible_outputs}")

    def test_punctuation_outside_quotes(self):
        """Test that punctuation is moved outside quotes."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {'punctuation_outside_quotes': 1.0},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = '"Hello, world!"'
        result = inserter._insert_punctuation_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        # Should move punctuation outside: "Hello, world"! or "Hello world"!
        self.assertTrue(
            '"Hello, world"!' in modified_text or '"Hello world"!' in modified_text,
            f"Should move punctuation outside quotes, got: {modified_text}"
        )

    def test_dropped_letters_error(self):
        """Test that letters are dropped from words."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {'dropped_letters': 1.0},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "Hello"
        result = inserter._insert_typography_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        self.assertLess(len(modified_text), len(input_text), "Should be shorter after dropping letter")
        # Should be one of: ello, Hllo, Helo, Hell
        possible_outputs = ['ello', 'Hllo', 'Helo', 'Hell']
        self.assertIn(modified_text, possible_outputs, f"Should drop one letter, got: {modified_text}")

    def test_swapped_letters_error(self):
        """Test that adjacent letters are swapped."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {'swapped_letters': 1.0},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "world"  # All unique characters
        result = inserter._insert_typography_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        self.assertEqual(len(modified_text), len(input_text), "Should maintain same length")
        # Should be one of: owrld, wrold, wolrd, wordl
        possible_outputs = ['owrld', 'wrold', 'wolrd', 'wordl']
        self.assertIn(modified_text, possible_outputs, f"Should swap adjacent letters, got: {modified_text}")

    def test_dropped_words_error(self):
        """Test that words are dropped while preserving formatting."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {'dropped_words': 1.0},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "The quick brown fox jumps"
        result = inserter._insert_typography_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        self.assertLess(len(modified_text.split()), len(input_text.split()), "Should have fewer words")

        # Should drop a middle word (not first "The" or last "jumps")
        possible_outputs = [
            "The brown fox jumps",    # dropped "quick"
            "The quick fox jumps",    # dropped "brown"
            "The quick brown jumps"   # dropped "fox"
        ]
        self.assertIn(modified_text, possible_outputs, f"Should drop middle word, got: '{modified_text}'")

    def test_spacing_around_punctuation(self):
        """Test that spaces are added before punctuation."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {'spaces_around_punctuation': 1.0},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "Hello, world!"
        result = inserter._insert_spacing_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        # Should add spaces before punctuation
        self.assertTrue(
            "Hello , world !" in modified_text,
            f"Should add spaces before punctuation, got: '{modified_text}'"
        )

    def test_missing_spaces_error(self):
        """Test that spaces between words are removed."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {'missing_spaces': 1.0},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "Hello world"
        result = inserter._insert_spacing_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        self.assertIn("Helloworld", modified_text, f"Should remove space between words, got: '{modified_text}'")

    def test_homophone_substitution(self):
        """Test that homophones are correctly substituted."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {'substitution_rate': 1.0},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "There are books"
        result = inserter._insert_homophone_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        # "There" should become "Their" or "They're"
        self.assertTrue(
            "Their are books" in modified_text or "They're are books" in modified_text,
            f"Should substitute 'There' with homophone, got: '{modified_text}'"
        )

    def test_homophone_capitalization_preserved(self):
        """Test that capitalization is preserved in homophone substitution."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {'substitution_rate': 1.0},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "There are books"  # Capital T
        result = inserter._insert_homophone_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        # Should preserve capitalization
        self.assertTrue(
            modified_text.startswith("Their ") or modified_text.startswith("They're "),
            f"Should preserve capitalization, got: '{modified_text}'"
        )

    def test_contraction_nt_dropped(self):
        """Test that n't is dropped from contractions."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {'dropped_nt': 1.0}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "I couldn't find it"
        result = inserter._insert_contraction_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        self.assertIn("I could find it", modified_text, f"Should drop n't, got: '{modified_text}'")

    def test_keyboard_neighbor_substitution(self):
        """Test that keyboard neighbors are correctly substituted."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {},
                'keyboard_neighbors': {'substitution_rate': 1.0}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=12345)

        input_text = "hello"  # 'h' has neighbors 'g' and 'j'
        result = inserter._insert_keyboard_neighbor_errors(input_text, nlp(input_text))
        modified_text, error_count = result

        self.assertGreater(error_count, 0, "Should apply at least one error")
        self.assertNotEqual(modified_text, input_text, "Should modify the text")
        self.assertEqual(len(modified_text), len(input_text), "Should maintain same length")
        # Verify only alphabetic characters are present
        self.assertTrue(modified_text.isalpha(), f"Should only contain letters, got: '{modified_text}'")

    def test_keyboard_neighbor_no_punctuation_substitution(self):
        """Test that keyboard neighbors NEVER substitute letters with punctuation."""
        # Create isolated config for just this error type
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {},
                'keyboard_neighbors': {'substitution_rate': 1.0}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        
        # Test specific edge keys that are adjacent to punctuation on QWERTY keyboard
        edge_key_tests = [
            ('l', ';'),  # 'l' is next to semicolon but should only become 'k'
            ('p', '['),  # 'p' is next to '[' but should only become 'o' 
            ('m', ','),  # 'm' is next to comma but should only become 'n'
        ]
        
        for target_letter, forbidden_punct in edge_key_tests:
            with self.subTest(letter=target_letter, punct=forbidden_punct):
                input_text = target_letter + "est"  # lest, pest, mest
                
                errors_applied = 0
                for seed in range(100):  # Try many seeds to force error application
                    inserter = SyntheticErrorInserter(config, seed=seed)
                    result = inserter._insert_keyboard_neighbor_errors(input_text, nlp(input_text))
                    modified_text, error_count = result
                    
                    if error_count > 0:
                        errors_applied += 1
                        # CRITICAL: Verify the forbidden punctuation is never introduced
                        self.assertNotIn(forbidden_punct, modified_text, 
                                       f"Letter '{target_letter}' was incorrectly replaced with punctuation '{forbidden_punct}': '{input_text}' -> '{modified_text}'")
                        # Verify only valid alphabetic substitution occurred
                        self.assertTrue(modified_text.isalpha(), 
                                      f"Non-alphabetic character introduced: '{input_text}' -> '{modified_text}'")
                        # Verify the substitution was actually in the target position
                        if modified_text != input_text:
                            self.assertNotEqual(modified_text[0], target_letter, 
                                              f"Target letter '{target_letter}' should have been changed")
                
                # Ensure we actually tested some error applications
                self.assertGreater(errors_applied, 0, f"No errors applied for letter '{target_letter}' - test is invalid")

    def test_keyboard_neighbor_capitalization_preserved(self):
        """Test that capitalization is preserved in keyboard neighbor substitution."""
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {},
                'keyboard_neighbors': {'substitution_rate': 1.0}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }

        # Test different capitalization patterns
        test_cases = [
            ("Hello", 0),      # Capital first letter
            ("hELLO", 0),      # Lowercase first, capitals after
            ("HeLLo", 2),      # Mixed case, target L at position 2
            ("WORLD", 1),      # All caps, target O at position 1
        ]
        
        for input_text, target_pos in test_cases:
            with self.subTest(input_text=input_text):
                original_char = input_text[target_pos]
                original_is_upper = original_char.isupper()
                
                substitutions_found = 0
                capitalization_preserved = 0
                
                # Try many seeds to force substitutions
                for seed in range(100):
                    inserter = SyntheticErrorInserter(config, seed=seed)
                    result = inserter._insert_keyboard_neighbor_errors(input_text, nlp(input_text))
                    modified_text, error_count = result
                    
                    if error_count > 0 and modified_text != input_text:
                        substitutions_found += 1
                        new_char = modified_text[target_pos]
                        new_is_upper = new_char.isupper()
                        
                        # CRITICAL: Verify capitalization pattern is preserved
                        if original_is_upper == new_is_upper:
                            capitalization_preserved += 1
                        
                        # Additional verification: rest of string unchanged
                        expected_unchanged = input_text[:target_pos] + input_text[target_pos+1:]
                        actual_unchanged = modified_text[:target_pos] + modified_text[target_pos+1:]
                        self.assertEqual(actual_unchanged, expected_unchanged, 
                                       f"Only position {target_pos} should change")
                
                self.assertGreater(substitutions_found, 0, f"No substitutions found for '{input_text}' - test invalid")
                
                # CRITICAL: ALL substitutions must preserve capitalization
                self.assertEqual(substitutions_found, capitalization_preserved, 
                               f"Capitalization not preserved in all cases: {capitalization_preserved}/{substitutions_found} for '{input_text}'")

    def test_keyboard_neighbor_edge_keys(self):
        """Test keyboard neighbor substitution for edge keys produces ONLY valid neighbors."""
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {},
                'keyboard_neighbors': {'substitution_rate': 1.0}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }

        # Test edge keys and their EXACT expected neighbors
        edge_cases = [
            ('q', ['w']),         # q has only one neighbor: w
            ('p', ['o']),         # p has only one neighbor: o  
            ('a', ['s']),         # a has only one neighbor: s
            ('l', ['k']),         # l has only one neighbor: k
            ('z', ['x']),         # z has only one neighbor: x
            ('m', ['n'])          # m has only one neighbor: n
        ]
        
        for target_letter, expected_neighbors in edge_cases:
            with self.subTest(letter=target_letter):
                input_text = target_letter + "est"  # qest, pest, aest, etc.
                
                actual_substitutions = set()
                errors_applied = 0
                
                # Try many seeds to collect all possible substitutions
                for seed in range(200):
                    inserter = SyntheticErrorInserter(config, seed=seed)
                    result = inserter._insert_keyboard_neighbor_errors(input_text, nlp(input_text))
                    modified_text, error_count = result
                    
                    if error_count > 0 and modified_text != input_text:
                        errors_applied += 1
                        # Extract the substituted character
                        substituted_char = modified_text[0]  # First character was the target
                        actual_substitutions.add(substituted_char)
                
                # Verify we actually tested some substitutions
                self.assertGreater(errors_applied, 0, f"No substitutions found for '{target_letter}' - test invalid")
                
                # CRITICAL: Verify ONLY the expected neighbors were used
                for actual_sub in actual_substitutions:
                    self.assertIn(actual_sub, expected_neighbors, 
                                f"Letter '{target_letter}' was incorrectly substituted with '{actual_sub}'. Expected only: {expected_neighbors}")
                
                # Verify that if we got any substitutions, they match the expected set
                if actual_substitutions:
                    unexpected = actual_substitutions - set(expected_neighbors)
                    self.assertEqual(len(unexpected), 0, 
                                   f"Unexpected substitutions for '{target_letter}': {unexpected}. Expected only: {expected_neighbors}")

    def test_preserve_ratio_functionality(self):
        """Test that preserve ratio works correctly."""
        # Config with 50% preserve ratio and one error type
        config = {
            'error_rates': {
                'punctuation': {'quote_spaces': 1.0},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.5,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=42)

        input_text = '"Hello world"'
        preserved_count = 0
        modified_count = 0

        # Run many times to test ratio
        for _ in range(100):
            result = inserter.insert_errors(input_text)
            if result == input_text:
                preserved_count += 1
            else:
                modified_count += 1

        # Should be roughly 50/50 split (allow some variance)
        self.assertGreater(preserved_count, 30, "Should preserve some examples")
        self.assertGreater(modified_count, 30, "Should modify some examples")

    def test_min_chunk_length_respected(self):
        """Test that minimum chunk length is respected."""
        config = {
            'error_rates': {
                'punctuation': {'quote_spaces': 1.0},
                'typography': {},
                'spacing': {},
                'homophones': {},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 1,
            'min_chunk_length': 50
        }
        inserter = SyntheticErrorInserter(config, seed=42)

        short_text = "Hi"  # Too short
        result = inserter.insert_errors(short_text)

        self.assertEqual(result, short_text, "Should preserve text shorter than min_chunk_length")

    def test_max_errors_per_chunk_respected(self):
        """Test that maximum errors per chunk is respected."""
        config = {
            'error_rates': {
                'punctuation': {},
                'typography': {},
                'spacing': {},
                'homophones': {'substitution_rate': 1.0},
                'contractions': {}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 2,
            'min_chunk_length': 1
        }
        inserter = SyntheticErrorInserter(config, seed=42)

        # Long text that could have many errors
        long_text = "There are their books over there and they're reading them."

        # Count errors by comparing with original
        original_words = set(long_text.split())
        result = inserter.insert_errors(long_text)
        result_words = set(result.split())

        # This is a rough check - in practice, we'd need more sophisticated error counting
        differences = len(original_words.symmetric_difference(result_words))
        self.assertLessEqual(differences, 4, "Should respect max errors per chunk")  # Allow some leeway


class TestErrorInserterIntegration(unittest.TestCase):
    """Integration tests for the complete error insertion workflow."""

    def test_multiple_error_types_can_coexist(self):
        """Test that multiple error types can be applied together."""
        config = {
            'error_rates': {
                'punctuation': {'quote_spaces': 0.5},
                'homophones': {'substitution_rate': 0.5},
                'typography': {'dropped_letters': 0.5},
                'spacing': {'spaces_around_punctuation': 0.5},
                'contractions': {'dropped_nt': 0.5}
            },
            'preserve_ratio': 0.0,
            'max_errors_per_chunk': 5,
            'min_chunk_length': 1
        }

        inserter = SyntheticErrorInserter(config, seed=42)
        input_text = '"There couldn\'t be any issues with this text!"'

        # Run multiple times and check that we can get different error combinations
        results = []
        for _ in range(20):
            result = inserter.insert_errors(input_text)
            results.append(result)

        # Should have variety in results
        unique_results = set(results)
        self.assertGreater(len(unique_results), 1, "Should produce varied error combinations")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)