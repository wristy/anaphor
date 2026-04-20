# Cell 1: Import necessary libraries
import sys
import os
from pathlib import Path
import pandas as pd
import openai
from typing import Dict, List, Tuple
import json
import google.generativeai as genai
import time
import ast
# Add the src directory to the path
sys.path.append(str(Path.cwd().parent / "src"))
from anaphor_comprehension_analyzer import AnaphorComprehensionAnalyzer

print("Libraries imported successfully!")
# Cell 2: Set up Google API (you'll need to add your API key)
# Set your Google API key# Or set it as an environment variable:
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key-here"




# Cell 3: Create the automated scoring class
class AutomatedScorer:
    """
    Automated scoring system using GPT-4 to evaluate comprehension answers.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the scorer with Google API key.
        
        Args:
            api_key: Google API key (if not provided, will use environment variable)
        """
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel('gemini-3.0-flash-preview')
        
        #response = model.generate_content("What is the capital of France?")
            
    
    def load_correct_answers(self, answers_file: str) -> Dict[Tuple[str, str, str], str]:
        """
        Load correct answers from CSV or Excel file.
        
        Args:
            answers_file: Path to the answers CSV or Excel file
            
        Returns:
            Dictionary mapping (passage, version, question) to correct answer
        """
        # Handle both CSV and Excel files
        if str(answers_file).endswith('.xlsx') or str(answers_file).endswith('.xls'):
            df = pd.read_excel(answers_file)
        else:
            df = pd.read_csv(answers_file)
        
        correct_answers = {}
        
        for _, row in df.iterrows():
            # Ensure passage is 2-digit string
            passage_num = str(row['Passage']).zfill(2)
            # Ensure version is uppercase string  
            version = str(row['Version']).strip().upper()
            # Ensure question is the right type (should be 10, 11, 12)
            question = row['Question']
            # Convert to int if it's a string, to ensure type consistency
            if isinstance(question, str):
                try:
                    question = int(question)
                except ValueError:
                    pass  # Keep as string if it can't be converted
            answer = str(row['Answer']).strip()
            
            key = (passage_num, version, question)
            correct_answers[key] = answer
        
        return correct_answers

    def load_study3_answers(self, answers_file: str = None) -> Dict[Tuple[int, int], str]:
        """
        Load correct answers for Study 3 from answers.csv file.
        
        Args:
            answers_file: Path to answers.csv file. If None, uses default path.
            
        Returns:
            Dictionary mapping (passage_num, question_num) to answer string.
            passage_num: 1-16
            question_num: 1-2 (first two questions only)
        """
        if answers_file is None:
            answers_file = Path(__file__).parent.parent / "study3_answers" / "answers.csv"
        else:
            answers_file = Path(answers_file)
        
        # Read the CSV file - it's a single row with all answers
        with open(answers_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Parse the CSV row - it contains quoted strings separated by commas
        # Format: 'van', 'gum', 'finger', ...
        try:
            # Use ast.literal_eval to safely parse the Python literal list
            answers_list = ast.literal_eval('[' + content + ']')
        except:
            # Fallback: manual parsing
            # Split by comma, then strip quotes from each item
            answers_list = []
            for item in content.split(','):
                item = item.strip().strip("'\"")
                answers_list.append(item)
        
        # Create mapping: (passage_num, question_num) -> answer
        # Order: Passage 1 Q1, Passage 1 Q2, Passage 2 Q1, Passage 2 Q2, ...
        correct_answers = {}
        answer_idx = 0
        
        for passage_num in range(1, 17):  # Passages 1-16
            for question_num in range(1, 3):  # Questions 1-2
                if answer_idx < len(answers_list):
                    correct_answers[(passage_num, question_num)] = answers_list[answer_idx].strip()
                    answer_idx += 1
        
        return correct_answers

    def load_study3_model_responses(self, model_response_file: str, 
                                    study3_path: str = None) -> pd.DataFrame:
        """
        Load Study 3 model responses from CSV and convert to long format.
        
        Handles two formats:
        1. Old format: Has metadata columns (study, version, passage_num, question_1, question_2, etc.)
        2. New format: Has Passage, Answer_Index, Answer columns
        
        Args:
            model_response_file: Name of the CSV file (e.g., "comprehension_llama3.csv" or "gpt2xl.csv")
            study3_path: Path to study3 directory. If None, uses default path.
            
        Returns:
            DataFrame in long format with columns: study, version, passage_num, passage_key,
            question_num, question_id, generated_answer, passage_text, question
        """
        if study3_path is None:
            study3_path = Path(__file__).parent.parent / "study3"
        else:
            study3_path = Path(study3_path)
        
        # Load the CSV
        results_path = Path(__file__).parent.parent / "results" / "model_responses" / "study 3" / model_response_file
        df = pd.read_csv(results_path)
        
        print(f"Loaded {len(df)} rows from {model_response_file}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if this is the old format (has metadata columns)
        has_old_format = 'study' in df.columns and 'version' in df.columns and 'question_1' in df.columns
        
        if has_old_format:
            print("Detected old format (with metadata columns)")
            return self._load_study3_old_format(df)
        else:
            print("Detected new format (Passage, Answer_Index, Answer)")
            return self._load_study3_new_format(df, study3_path)
    
    def _load_study3_old_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load Study 3 responses in old format (with metadata columns).
        
        Args:
            df: DataFrame with columns: study, version, passage_num, question_1, question_2, etc.
            
        Returns:
            DataFrame in long format
        """
        long_results = []
        
        for _, row in df.iterrows():
            # Parse questions list if it's a string
            questions = row.get('questions', [])
            if isinstance(questions, str):
                import ast
                try:
                    questions = ast.literal_eval(questions)
                except:
                    questions = []
            
            # Only process first 2 questions
            for q_num in [1, 2]:
                question_col = f'question_{q_num}'
                if question_col in row:
                    passage_num = int(row['passage_num']) if isinstance(row['passage_num'], str) else row['passage_num']
                    
                    long_row = {
                        'study': row.get('study', 'study3'),
                        'version': str(row['version']).strip().upper(),
                        'passage_num': passage_num,
                        'passage_key': str(passage_num).zfill(2),
                        'question_num': q_num,
                        'question_id': q_num,  # For study3, question_id = question_num
                        'question': questions[q_num - 1] if len(questions) >= q_num else "",
                        'generated_answer': str(row[question_col]).strip(),
                        'passage_text': row.get('passage_text', ''),
                        'passage_file': row.get('passage_file', f"{passage_num}.txt")
                    }
                    long_results.append(long_row)
        
        results_df = pd.DataFrame(long_results)
        print(f"Converted to long format: {len(results_df)} rows")
        print(f"Passages: {results_df['passage_num'].nunique()}, Versions: {results_df['version'].nunique()}")
        
        return results_df
    
    def _load_study3_new_format(self, df: pd.DataFrame, study3_path: Path) -> pd.DataFrame:
        """
        Load Study 3 responses in new format (Passage, Answer_Index, Answer).
        
        Args:
            df: DataFrame with columns: Passage, Answer_Index, Answer
            study3_path: Path to study3 directory
            
        Returns:
            DataFrame in long format
        """
        # Read questions and passage text from study3 passage files
        def read_study3_passage_data(passage_num: int) -> Dict:
            """Read questions and passage text from a study3 passage file."""
            passage_file = study3_path / f"{passage_num}.txt"
            if not passage_file.exists():
                return {'questions': [], 'passage_text': ''}
            
            with open(passage_file, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n') for line in f.readlines()]
            
            # Questions are on lines 7-10 (0-indexed: 7, 8, 9, 10)
            questions = []
            if len(lines) >= 11:
                questions = [lines[7], lines[8], lines[9], lines[10]]
            
            # Passage text is from line 11 onwards (body text)
            passage_text = ' '.join([line.strip() for line in lines[11:]]) if len(lines) > 11 else ''
            
            return {'questions': questions, 'passage_text': passage_text}
        
        # Cache passage data to avoid reading files multiple times
        passage_data_cache = {}
        
        # Convert to long format
        long_results = []
        versions = ['A', 'B', 'C', 'D']
        
        for _, row in df.iterrows():
            csv_passage = int(row['Passage'])
            answer_index = int(row['Answer_Index'])
            answer = str(row['Answer']).strip()
            
            # Convert CSV passage number to actual passage number and version
            # Passage 1-4 -> Passage 1, Versions A-D
            # Passage 5-8 -> Passage 2, Versions A-D
            # etc.
            actual_passage_num = ((csv_passage - 1) // 4) + 1
            version_idx = (csv_passage - 1) % 4
            version = versions[version_idx]
            
            # Only process first 2 questions
            if answer_index <= 2:
                # Get passage data (cached)
                if actual_passage_num not in passage_data_cache:
                    passage_data_cache[actual_passage_num] = read_study3_passage_data(actual_passage_num)
                
                passage_info = passage_data_cache[actual_passage_num]
                questions = passage_info['questions']
                question_text = questions[answer_index - 1] if len(questions) >= answer_index else ""
                
                long_row = {
                    'study': 'study3',
                    'version': version,
                    'passage_num': actual_passage_num,
                    'passage_key': str(actual_passage_num).zfill(2),
                    'question_num': answer_index,
                    'question_id': answer_index,  # For study3, question_id = question_num (1 or 2)
                    'question': question_text,
                    'generated_answer': answer,
                    'passage_text': passage_info['passage_text'],
                    'passage_file': f"{actual_passage_num}.txt"
                }
                long_results.append(long_row)
        
        results_df = pd.DataFrame(long_results)
        print(f"Converted to long format: {len(results_df)} rows")
        print(f"Passages: {results_df['passage_num'].nunique()}, Versions: {results_df['version'].nunique()}")
        
        return results_df

    def score_study3_answers(self, results_df: pd.DataFrame, 
                            correct_answers: Dict[Tuple[int, int], str]) -> pd.DataFrame:
        """
        Score Study 3 answers using batch processing by passage.
        
        Args:
            results_df: DataFrame with model responses in long format
            correct_answers: Dictionary mapping (passage_num, question_num) to correct answer
            
        Returns:
            DataFrame with scored results
        """
        scored_results = []
        
        # Group by passage and version
        for (passage_key, version), group in results_df.groupby(['passage_key', 'version']):
            print(f"Scoring passage {passage_key} version {version} ({len(group)} questions)...")
            
            # Get passage number as int for correct answer lookup
            passage_num = int(passage_key)
            
            # Build batch prompt
            questions_text = ""
            for idx, row in group.iterrows():
                question_id = row['question_id']
                question_text = row['question']
                generated_answer = row['generated_answer']
                # Correct answers don't have version, so use (passage_num, question_id)
                correct_answer = correct_answers.get((passage_num, question_id), "Unknown")
                
                questions_text += f"""
    Question {question_id}: {question_text}
    Correct Answer: {correct_answer}
    Generated Answer: {generated_answer}

    """
            
            # Get passage text from DataFrame (should be in first row)
            passage_text = group.iloc[0].get('passage_text', '') if len(group) > 0 else ""
            
            prompt = f"""
    You are an expert evaluator for reading comprehension questions, the correct answers to which are given to you. Your job is to evaluate a set of responses to those questions, by comparing those responses to the given correct answers.
    Use the passage for context to help you evaluate how well the responses match the correct answers but do not come up with your own answers.

    PASSAGE CONTEXT: {passage_text[:500]}...

    {questions_text}

    For each question, provide:
    1. A score of 1 if the answer matches the correct answer, 0 if incorrect
    2. A brief explanation of your reasoning

    Respond in JSON format with an array of results:
    [
        {{"question_id": 1, "score": 0, "reasoning": "explanation", "is_correct": false}},
        {{"question_id": 2, "score": 1, "reasoning": "explanation", "is_correct": true}}
    ]
    """

            try:
                response = self.model.generate_content(contents=prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                # Clean up response
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    if end != -1:
                        response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.find("```", start)
                    if end != -1:
                        response_text = response_text[start:end].strip()
                
                results = json.loads(response_text)
                
                # Map results back to DataFrame
                scored_data = group.copy()
                for result in results:
                    question_id = result['question_id']
                    mask = scored_data['question_id'] == question_id
                    if mask.any():
                        scored_data.loc[mask, 'gpt4_score'] = result['score']
                        scored_data.loc[mask, 'gpt4_reasoning'] = result['reasoning']
                        scored_data.loc[mask, 'is_correct'] = result['is_correct']
                        scored_data.loc[mask, 'correct_answer'] = correct_answers.get((passage_num, question_id), "")
                
                scored_results.append(scored_data)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error scoring passage {passage_key} version {version}: {e}")
                # Fall back to individual scoring or mark as error
                error_data = group.copy()
                error_data['gpt4_score'] = 0
                error_data['gpt4_reasoning'] = f"Error: {str(e)}"
                error_data['is_correct'] = False
                error_data['correct_answer'] = ""
                scored_results.append(error_data)
        
        if scored_results:
            return pd.concat(scored_results, ignore_index=True)
        else:
            return pd.DataFrame()




    def score_answer(self, question: str, correct_answer: str, generated_answer: str, 
                    passage_text: str = None) -> Dict[str, any]:
        """
        Use gemini flash to score a single answer.
        
        Args:
            question: The question being asked
            correct_answer: The correct answer
            generated_answer: The answer generated by the model
            passage_text: Optional passage text for context
            
        Returns:
            Dictionary with score and reasoning
        """
        
        # Create the scoring prompt
    
        prompt = f"""
You are an expert evaluator for reading comprehension questions, grading some answers. Your task is to determine if the given answer is correct based on the correct answer.

QUESTION: {question}

CORRECT ANSWER: {correct_answer}

GIVEN ANSWER: {generated_answer}

{f"PASSAGE CONTEXT: {passage_text[:500]}..." if passage_text else ""}

Please evaluate the generated answer and provide:
1. A score of 1 if the answer is correct, 0 if incorrect, no other values.
2. A brief explanation of your reasoning

Respond in JSON format:
{{
    "score": <0 or 1>,
    "reasoning": "<brief explanation>",
    "is_correct": <true or false>
}}
"""

        try:
            # response = openai.ChatCompletion.create(
            #     model="gpt-4",
            #     messages=[
            #         {"role": "system", "content": "You are an expert evaluator for reading comprehension questions. Always respond in valid JSON format."},
            #         {"role": "user", "content": prompt}
            #     ],
            #     temperature=0.1,
            #     max_tokens=200
            # )
            
            # response = openai.responses.create(
            #     model="gpt-4o",
            #     instructions="You are an expert evaluator for reading comprehension questions. Always respond in valid JSON format",
            #     input= f"{prompt}"
            # )
            

            # response = self.model.generate_content(
            #     contents=f"You are an expert evaluator for reading comprehension questions. Always respond in valid JSON format. Here is the prompt: {prompt}",
            # )
            # print("llm response: ", response.text)

            # # response_text = response.text if hasattr(response, 'text') else str(response)
            # # result = json.loads(response_text)
            # return response.text
        # In src/automated_scorer.py, replace the response handling section with:

            response = self.model.generate_content(
                contents=f"You are an expert evaluator for reading comprehension questions. Always respond in valid JSON format. Here is the prompt: {prompt}",
            )

            # Debug the response
            response_text = response.text if hasattr(response, 'text') else str(response)
            print(f"llm response: {response_text}")

            # Check if response is empty
            if not response_text or response_text.strip() == "":
                print("Warning: Empty response from API")
                return {
                    "score": 0,
                    "reasoning": "Empty response from API",
                    "is_correct": False
                }

            # Clean up the response - remove markdown code blocks if present
            if "```json" in response_text:
                # Extract JSON from markdown code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                # Extract JSON from generic code block
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()

            # Try to parse JSON
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as json_err:
                print(f"JSON decode error: {json_err}")
                print(f"Cleaned response was: '{response_text}'")
                return {
                    "score": 0,
                    "reasoning": f"Invalid JSON response: {str(json_err)}",
                    "is_correct": False
                }
                            
            
        except Exception as e:
            print(f"Error scoring answer: {e}")
            return {
                "score": 0,
                "reasoning": f"Error in scoring: {str(e)}",
                "is_correct": False
            }
    
    # Updated score_all_answers method in automated_scorer.py

    def score_passage_batch(self, passage_data: pd.DataFrame, correct_answers: Dict[Tuple[str, str, str], str]) -> pd.DataFrame:
        """
        Score all questions for a single passage in one API call.
        
        Args:
            passage_data: DataFrame with all questions for one passage
            correct_answers: Dictionary of correct answers
            
        Returns:
            DataFrame with scored results
        """
        if len(passage_data) == 0:
            return passage_data
        
        # Get passage info
        passage_num = passage_data.iloc[0]['passage_key']
        version = passage_data.iloc[0]['version']
        passage_text = passage_data.iloc[0]['passage_text']
        
        # Build batch prompt
        questions_text = ""
        for idx, row in passage_data.iterrows():
            question_id = row['question_id']
            question_text = row['question']
            generated_answer = row['generated_answer']
            correct_key = (passage_num, version, question_id)
            correct_answer = correct_answers.get(correct_key, "Unknown")
            
            questions_text += f"""
    Question {question_id}: {question_text}
    Correct Answer: {correct_answer}
    Generated Answer: {generated_answer}

    """
        
        prompt = f"""
    You are an expert evaluator for reading comprehension questions, the correct answers to which are given to you. Your job is to evalute a set of responses to those questions, by comparing those responses to the given correct answers.
    Use the passge for context to help you evaluate how well the responses match the correct answers but do not come up with your own answers.

    PASSAGE CONTEXT: {passage_text[:500]}...

    {questions_text}

    For each question, provide:
    1. A score of 1 if the answer matches the correct answer, 0 if incorrect
    2. A brief explanation of your reasoning

    Respond in JSON format with an array of results:
    [
        {{"question_id": 10, "score": 0, "reasoning": "explanation", "is_correct": false}},
        {{"question_id": 11, "score": 1, "reasoning": "explanation", "is_correct": true}},
        {{"question_id": 12, "score": 0, "reasoning": "explanation", "is_correct": false}}
    ]
    """

        try:
            response = self.model.generate_content(contents=prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Clean up response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            
            results = json.loads(response_text)
            
            # Map results back to DataFrame
            scored_data = passage_data.copy()
            for result in results:
                question_id = result['question_id']
                mask = scored_data['question_id'] == question_id
                if mask.any():
                    scored_data.loc[mask, 'gpt4_score'] = result['score']
                    scored_data.loc[mask, 'gpt4_reasoning'] = result['reasoning']
                    scored_data.loc[mask, 'is_correct'] = result['is_correct']
                    scored_data.loc[mask, 'correct_answer'] = correct_answers.get((passage_num, version, question_id), "")
            
            return scored_data
            
        except Exception as e:
            print(f"Error scoring passage {passage_num}: {e}")
            # Fall back to individual scoring
            error_data = passage_data.copy()
            error_data['gpt4_score'] = 0
            error_data['gpt4_reasoning'] = f"Error: {str(e)}"
            error_data['is_correct'] = False
            error_data['correct_answer'] = ""
            return error_data

    def score_all_answers(self, results_df: pd.DataFrame, correct_answers: Dict[Tuple[str, str, str], str]) -> pd.DataFrame:
        """
        Score all answers using batch processing by passage.
        """
        scored_results = []
        
        # Group by passage
        for (passage_key, version), group in results_df.groupby(['passage_key', 'version']):
            print(f"Scoring passage {passage_key} version {version} ({len(group)} questions)...")
            scored_group = self.score_passage_batch(group, correct_answers)
            scored_results.append(scored_group)
            print("scored result: ", scored_group)
            time.sleep(0.5) 
        if scored_results:
            return pd.concat(scored_results, ignore_index=True)
        else:
            return pd.DataFrame()  #
        


    def calculate_passage_accuracy(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate accuracy for each passage by aggregating question scores.
        
        Args:
            scored_df: DataFrame with scored individual questions
            
        Returns:
            DataFrame with passage-level accuracy
        """
        # Group by passage and calculate accuracy
        passage_accuracy = scored_df.groupby(['study', 'version', 'passage_num', 'passage_file']).agg({
            'gpt4_score': ['mean', 'sum', 'count'],
            'is_correct': ['sum', 'count'],
            'passage_text': 'first'  # Take first occurrence of passage text
        }).reset_index()
        
        # Flatten column names
        passage_accuracy.columns = [
            'study', 'version', 'passage_num', 'passage_file', 
            'avg_score', 'total_score', 'num_questions',
            'correct_answers', 'total_answers', 'passage_text'
        ]
        
        # Calculate accuracy percentage
        passage_accuracy['accuracy'] = passage_accuracy['correct_answers'] / passage_accuracy['total_answers']
        
        return passage_accuracy

    def save_results(self, df: pd.DataFrame, output_path: str):
        """
        Save results to CSV and Excel files.
        
        Args:
            df: Results DataFrame
            output_path: Base path for output files
        """
        # Save as CSV
        csv_path = f"{output_path}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Save as Excel
        excel_path = f"{output_path}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Results saved to {excel_path}")
            