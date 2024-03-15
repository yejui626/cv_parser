# Resume Parser

## Overview
This Resume Parser is a Python-based application that is designed to extract relevant candidate information from resumes, including personal details, educational background, work experience, skills, and more. The parser utilizes advanced natural language processing (NLP) techniques, including OpenAI's GPT models and traditional NLP libraries, to accurately extract and parse candidate information from resumes.

## Features
- **Generative AI Based Parsing**: Leveraging OpenAI's GPT models, the parser is capable of accurately extracting candidate information from resumes.
- **Automated Extraction**: The parser automates the extraction of candidate details, including name, age, contact information, education background, work experience, skills, and more.
- **Customizable Criteria Evaluation**: The application allows for the customization of evaluation criteria based on specific job requirements and preferences.
- **Applicant Tracking System Integration**: The parsed candidate information can be seamlessly integrated into an Applicant Tracking System (ATS) for efficient management of the recruitment process.

## Usage
1. **Parsing**: Utilize the `parse_pdf` method to parse individual PDF resumes and extract candidate information.
2. **Evaluation**: Evaluate the extracted candidate information based on predefined criteria using the `evaluate_candidate` method.
3. **Asynchronous Parsing**: For batch processing of multiple PDF resumes, utilize the `parsing` method, which asynchronously parses all resumes in the specified directory.
4. **Result Export**: Export the parsed candidate information to a DataFrame for further analysis or integration with an ATS.

## Example Output
You may access to the [Example Output](https://github.com/yejui626/cv_parser/files/14612326/CVParser_2024-03-15_150224.xlsx) that includes the extracted information for the candidate, the information for the selected criteria and the evaluation results based on the assigned weightage.

## Dependencies
- Python 3.8
- OpenAI GPT models (gpt-3.5-turbo-0125 & text-embeddings-ada-002)
- PyPDF2
- Pandas
- NumPy
- Flask
- spaCy
- LangChain

