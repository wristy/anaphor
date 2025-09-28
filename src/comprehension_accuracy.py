from litellm import completion
import os
from typing import List, Dict
import torch
from pathlib import Path
import pandas as pd
import numpy as np

class AnaphorComprehensionAnalyzer:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def ask_questions(self, context: str, questions: List[str]) -> Dict[str, str]:
        answers = {}
        
        for i, question in enumerate(questions, 1):
            prompt = f"Context: {context}\n\nQuestion: {question.strip()}\nAnswer:"
            
            response = completion(
                model=self.model_name,  # e.g., "gpt-3.5-turbo", "mistral/mistral-7b-instruct"
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            answers[f"question_{i}"] = answer
            
        return answers
    
    def read_passage(self, file_path: str) -> str:
        """
        Read and process a passage from Study 1 format.
        
        Args:
            file_path: Path to the passage file (e.g., p01A.txt)
            
        Returns:
            Processed passage text with anaphor
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Extract text content (skip the first column which contains numbers)
        text_lines = []
        for line in lines:
            if '\t' in line:
                # Split on tab and take everything after the first column
                text_content = line.split('\t', 1)[1].strip()
                if text_content:
                    text_lines.append(text_content)
        
        # Join lines and clean up
        passage = ' '.join(text_lines)
        return ' '.join(passage.split())  # Remove extra whitespace
    
    def read_questions(self, file_path: str) -> List[str]:
        """
        Read questions from Study 1 format.
        
        Args:
            file_path: Path to the questions file (e.g., q01.txt)
            
        Returns:
            List of question strings
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        questions = []
        for line in lines:
            if '\t' in line:
                # Split on tab and take everything after the first column
                question = line.split('\t', 1)[1].strip()
                if question:
                    questions.append(question)
        
        return questions
    
    def make_prompt(self, story: str, questions: List[str]) -> str:
        """
        Create a prompt for the LLM in Study 1 format.
        
        Args:
            story: The passage text
            questions: List of questions about the passage
            
        Returns:
            Formatted prompt string
        """
        prompt = f"TEXT:\n{story}\n\nQUESTIONS:\n"
        for i, question in enumerate(questions, 1):
            prompt += f"{i}. {question}\n"
        prompt += "\nANSWERS:\n"
        return prompt
    
    def ask_questions(self, context: str, questions: List[str]) -> Dict[str, str]:
        """
        Generate answers for questions based on context.
        
        Args:
            context: The passage context
            questions: List of questions
            
        Returns:
            Dictionary mapping question numbers to answers
        """
        answers = {}
        
        for i, question in enumerate(questions, 1):
            # Create individual prompt for each question
            prompt = f"Context: {context}\n\nQuestion: {question.strip()}\nAnswer:"
            
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):].strip()
            
            # Clean up answer (take first sentence or first line)
            if '\n' in answer:
                answer = answer.split('\n')[0]
            if '.' in answer and len(answer.split('.')) > 1:
                answer = answer.split('.')[0] + '.'
            
            answers[f"question_{i}"] = answer
            print(f"Question {i}: {question}")
            print(f"Answer: {answer}\n")
        
        return answers
    
    def process_study(self, study_path: str, versions: List[str] = None) -> pd.DataFrame:
        """
        Process all passages in Study.
        
        Args:
            study_path: Base path to Study  directory
            versions: List of versions to process (e.g., ["A", "B", "C", "D"])
            
        Returns:
            DataFrame with all results
        """
        if versions is None:
            versions = ["A", "B", "C", "D"]
        
        results = []
        study_dir = Path(study_path)
        passages_dir = study_dir / "1 materials" / "passages" / "40 column versions"
        questions_dir = study_dir / "1 materials" / "passages" / "questions"
        
        # Find all passage files
        for version in versions:
            passage_files = sorted(passages_dir.glob(f"p*{version}.txt"))
            
            for passage_file in passage_files:
                # Extract passage number
                passage_num = passage_file.stem[1:3]  # e.g., "01" from "p01A.txt"
                
                # Read passage
                passage_text = self.read_passage(passage_file)
                
                # Read questions
                question_file = questions_dir / f"q{passage_num}.txt"
                if not question_file.exists():
                    print(f"Warning: No questions found for passage {passage_num}")
                    continue
                
                questions = self.read_questions(question_file)
                
                # Generate answers
                answers = self.ask_questions(passage_text, questions)
                
                # Store results
                result = {
                    "study": "study1",
                    "version": version,
                    "passage_num": passage_num,
                    "passage_file": passage_file.name,
                    "passage_text": passage_text,
                    "questions": questions,
                    **answers
                }
                results.append(result)
                
                print(f"Processed Study 1 - Version {version} - Passage {passage_num}")
        
        return pd.DataFrame(results)
    
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


def main():
    """
    Main function to run Study 1 analysis.
    """
    # Configuration
    models = [
        "openai-community/gpt2-xl",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1"
    ]
    
    study1_path = "/Users/varshinichinta/Desktop/anaphor/study 1 (context)"
    
    # Process each model
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Processing Study 1 with model: {model_name}")
        print(f"{'='*60}")
        
        try:
            analyzer = AnaphorComprehensionAnalyzer(model_name)
            
            # Process Study 1
            breakpoint()
            results = analyzer.process_study1(study1_path)

            
            # Save results
            model_safe_name = model_name.replace('/', '_')
            output_path = f"study1_results_{model_safe_name}"
            analyzer.save_results(results, output_path)
            
            print(f"\nCompleted processing {len(results)} passages with {model_name}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue


if __name__ == "__main__":
    main()