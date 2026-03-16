"""
Interactive script to add questions to the evaluation dataset.
"""

import json
import argparse
from pathlib import Path


def load_dataset(filepath: str) -> dict:
    """Load existing dataset"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_dataset(dataset: dict, filepath: str):
    """Save dataset to file"""
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)


def get_next_id(dataset: dict) -> int:
    """Get next available question ID"""
    if not dataset['questions']:
        return 1
    return max(q['id'] for q in dataset['questions']) + 1


def add_question_interactive(dataset: dict):
    """Add a question through interactive prompts"""
    print("\n" + "="*60)
    print("Add New Question to Evaluation Dataset")
    print("="*60)
    
    question_id = get_next_id(dataset)
    
    # Get question text
    print(f"\nQuestion ID: {question_id}")
    question_text = input("Question text: ").strip()
    
    if not question_text:
        print("Error: Question cannot be empty")
        return False
    
    # Get reference answer
    reference_answer = input("Reference answer: ").strip()
    
    if not reference_answer:
        print("Error: Reference answer cannot be empty")
        return False
    
    # Get category
    print("\nCategory options: definitional, factual, procedural, comparison, out_of_domain")
    category = input("Category: ").strip().lower()
    
    if not category:
        category = "factual"
    
    # Get difficulty
    print("\nDifficulty options: easy, medium, hard")
    difficulty = input("Difficulty [medium]: ").strip().lower()
    
    if not difficulty:
        difficulty = "medium"
    
    # In document?
    in_doc_input = input("Is answer in document? (y/n) [y]: ").strip().lower()
    in_document = in_doc_input != 'n'
    
    # Create question object
    new_question = {
        "id": question_id,
        "question": question_text,
        "reference_answer": reference_answer,
        "category": category,
        "difficulty": difficulty,
        "in_document": in_document,
        "requires_retrieval": in_document
    }
    
    # Preview
    print("\n" + "-"*60)
    print("Preview:")
    print(json.dumps(new_question, indent=2))
    print("-"*60)
    
    # Confirm
    confirm = input("\nAdd this question? (y/n) [y]: ").strip().lower()
    
    if confirm == 'n':
        print("Question not added.")
        return False
    
    # Add to dataset
    dataset['questions'].append(new_question)
    dataset['metadata']['total_questions'] = len(dataset['questions'])
    
    print(f"\n✓ Question {question_id} added successfully!")
    return True


def add_question_from_args(dataset: dict, args):
    """Add a question from command-line arguments"""
    question_id = get_next_id(dataset)
    
    new_question = {
        "id": question_id,
        "question": args.question,
        "reference_answer": args.answer,
        "category": args.category,
        "difficulty": args.difficulty,
        "in_document": args.in_document,
        "requires_retrieval": args.in_document
    }
    
    dataset['questions'].append(new_question)
    dataset['metadata']['total_questions'] = len(dataset['questions'])
    
    print(f"✓ Question {question_id} added successfully!")
    return True


def list_questions(dataset: dict):
    """List all questions in the dataset"""
    print("\n" + "="*60)
    print(f"Evaluation Dataset ({len(dataset['questions'])} questions)")
    print("="*60)
    
    for q in dataset['questions']:
        in_doc = "✓" if q['in_document'] else "✗"
        print(f"\n[{q['id']:2d}] {in_doc} ({q['category']}, {q['difficulty']})")
        print(f"Q: {q['question'][:70]}...")
        print(f"A: {q['reference_answer'][:70]}...")


def main():
    parser = argparse.ArgumentParser(description="Manage evaluation dataset questions")
    parser.add_argument("--dataset", type=str, 
                        default="evaluation/dataset/evaluation_dataset.json",
                        help="Path to dataset JSON file")
    parser.add_argument("--list", action="store_true",
                        help="List all questions")
    
    # Arguments for non-interactive mode
    parser.add_argument("--question", type=str,
                        help="Question text")
    parser.add_argument("--answer", type=str,
                        help="Reference answer")
    parser.add_argument("--category", type=str, default="factual",
                        help="Question category")
    parser.add_argument("--difficulty", type=str, default="medium",
                        help="Question difficulty")
    parser.add_argument("--in-document", action="store_true",
                        help="Answer is in document")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    dataset = load_dataset(str(dataset_path))
    
    # List mode
    if args.list:
        list_questions(dataset)
        return
    
    # Non-interactive mode
    if args.question and args.answer:
        add_question_from_args(dataset, args)
        save_dataset(dataset, str(dataset_path))
        print(f"✓ Dataset saved to {dataset_path}")
        return
    
    # Interactive mode
    print("Interactive Question Entry Mode")
    print("(Press Ctrl+C to exit)\n")
    
    try:
        while True:
            if add_question_interactive(dataset):
                save_dataset(dataset, str(dataset_path))
                print(f"✓ Dataset saved to {dataset_path}")
            
            # Ask if want to add another
            another = input("\nAdd another question? (y/n) [n]: ").strip().lower()
            if another != 'y':
                break
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    print(f"\nFinal dataset has {len(dataset['questions'])} questions")


if __name__ == "__main__":
    main()
