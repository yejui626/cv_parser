import pandas as pd
from flask import Flask, request, render_template, send_file, jsonify, make_response
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import secrets
import string

def parse_criteria_excel(file_path):
    """
    Parse the criteria Excel file and perform validations on the input data.
    1. Preprocessing of criteria Excel file.
    2. Check if all criteria column names follows the format.
    3. Check if the selected criteria's input is not empty.
    4. Format validation on all criteria's input.

    Parameters:
    file_path (str): The file path of the criteria Excel file.

    Returns:
    str or None: If there are validation errors, returns a JSON string containing the error message.
                If there are no validation errors, returns None.
    """
    batch_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
    # Preprocessing of the criteria Excel file
    criteria_df = pd.read_excel(file_path,skiprows=4,header=1,skipfooter=1)
    ## List of specified column names
    specified_columns = ['Selected', 'Criterias', 'Priority', 'Weightage', 'Input']
    ## Drop columns that are not in the specified list
    criteria_df = criteria_df.drop(columns=criteria_df.columns.difference(specified_columns))
    ## Set "Criterias" column as the index
    criteria_df.set_index('Criterias', inplace=True)

    # Define the expected index values
    expected_index_values = [
        'Education Background', 'Academic Result (CGPA)', 'Skill Groups',
        'Years of Total Work Experience', 'Match Phrase', 'Technology (Tools, Program, System)',
        'Years of experience in similar role', 'Years of experience in exact role' ,
        'Professional Certificate', 
        'Candidate Current Location', 'Targeted Employer', 'Age', 'Language',
        'Expected Salary in RM', 'Years of Graduation'
    ]

    # # Check if all expected index values are contained within the DataFrame index
    # for index in expected_index_values:
    #     if not index in criteria_df.index or index == 'Years of experience in similar role' or index =='Years of experience in exact role':
    #         return render_template('main.html', batch_token=batch_token, error_status=f"The index of 'criteria_df' does not match the expected format. Please do not modify the names for the criteria column!")
    #     else:
    #         print("The index of 'criteria_df' matches the expected format.")
    # Check if all expected index values are contained within the DataFrame index
    for index in criteria_df.index :
        if not index in expected_index_values:
            print (f"NOTTTT {index}")
            return render_template('main.html', batch_token=batch_token, error_status=f"The index of 'criteria_df' does not match the expected format. Please do not modify the names for the criteria column!")
        else:
            print("The index of 'criteria_df' matches the expected format.")

    # Replace NaN values in 'Input' column with "N/A"
    criteria_df['Input'].fillna('N/A', inplace=True)
    # Parse 'Selected' column into True and False
    criteria_df['Selected'] = criteria_df['Selected'].apply(lambda x: True if '✔' in str(x) else False)

    # Check if 'Selected' is True but 'Input' is 'N/A'
    for index in criteria_df.index:
        if (criteria_df['Selected'][index] == True) and (criteria_df['Input'][index] == 'N/A'):
            return render_template('main.html', batch_token=batch_token, error_status=f"Selected is True but Input is empty. Please provide valid input for such rows.:{index}")
        else:
            print("No error found: 'Selected' is not True while 'Input' is 'N/A' for any row in the DataFrame.",index)
    
    # Validation for Education background
    # No validation needed

    # Validation for Academic Results (CGPA)
    try:
        temp = criteria_df['Input']['Academic Result (CGPA)']
        if temp == 'N/A':
            temp = 0
        float_value = float(temp)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Please make sure you only input digits for CGPA")
        

    # Validation for Skill Group
    try:
        values = [value.strip() for value in criteria_df['Input']["Skill Groups"].split(',')]
        if values:
            print(values)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Skill Groups'. Input should follow the format: '[Value], [Value], [Value]'")

    # Validation for Years of Experience
    try:
        input_value = criteria_df['Input']['Years of Total Work Experience']
        if input_value == 'N/A':
            input_value = "0"
        # Define the expected format patterns
        expected_format_patterns = [
            r'\d+$',  # Option 1: Number
            r'\d+-\d+$',  # Option 2: Number-Number
            r'>\d+$',  # Option 3: >Number
            r'<\d+$'  # Option 4: <Number
        ]
        
        # Check if the input matches any of the expected format patterns
        if not any(re.match(pattern, input_value) for pattern in expected_format_patterns):
            return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Years of Total Work Experience'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Years of Total Work Experience'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")

    # Validation for Match Phrase
    try:
        values = [value.strip() for value in criteria_df['Input']['Match Phrase'].split(',')]
        if values:
            print(values)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Match Phrase'. Input should follow the format: '[Value], [Value], [Value]'")

    # Validation for Technology
    try:
        values = [value.strip() for value in criteria_df['Input']['Technology (Tools, Program, System)'].split(',')]
        if values:
            print(values)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Technology (Tools, Program, System)'. Input should follow the format: '[Value], [Value], [Value]'")

    # Validation for Years of Experience in Role
    try:
        if 'Years of experience in similar role' in criteria_df.index:
            input_value = criteria_df['Input']['Years of experience in similar role']
            if input_value == 'N/A':
                input_value = "0"
        else:
            input_value = criteria_df['Input']['Years of experience in exact role']
            if input_value == 'N/A':
                input_value = "0"
        
        # Define the expected format patterns
        expected_format_patterns = [
            r'\d+$',  # Option 1: Number
            r'\d+-\d+$',  # Option 2: Number-Number
            r'>\d+$',  # Option 3: >Number
            r'<\d+$'  # Option 4: <Number
        ]
        
        # Check if the input matches any of the expected format patterns
        if not any(re.match(pattern, input_value) for pattern in expected_format_patterns):
            return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Years of Experience in role'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Years of Experience in role'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")

    # # Validation for Local/Expat
    # try:
    #     if criteria_df['Input']['Local/Expat'] != "N/A":
    #         if not (criteria_df['Input']['Local/Expat'].lower().strip()) == 'local' or (criteria_df['Input']['Local/Expat'].lower().strip()=='expat'):
    #             return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Local/Expat'. Input should follow the format: 'Local/Expat from dropdown.'")
    # except:
    #     return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Local/Expat'. Input should follow the format: 'Local/Expat from dropdown.'")


    # Validation for Professional Certificate
    try:
        values = [value.strip() for value in criteria_df['Input']['Professional Certificate'].split(',')]
        if values:
            print(values)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Professional Certificate'. Input should follow the format: '[Value], [Value], [Value]'")
    
    # Validation for Candidate Current Location
    try:
        values = [value.strip() for value in criteria_df['Input']['Candidate Current Location'].split(',')]
        if values:
            print(values)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Candidate Current Location'. Input should follow the format: 'Option 1: Country, Option 2 : State, Country, Option 3 : City, State, Country'")


    def validate_input_format(input_string): 
        """
            Check if CVMATCHING template format correct for 
            Example: 
                True - "include(Shell, BP) ,  exclude( KLCC, Novella Clinical, Fidelity Investments)    
                True - "include(), exclude()"   
                True - "include(Shell, BP) , exclude()"    
                True - "include() , exclude(Shell, BP)" 
                False -  "include() , exclude(Shell, " 
        """
        # Regular expression pattern to match the valid format
        # pattern = r'^(include\([\w\s,]*\)\s*,\s*exclude\([\w\s,]*\)\s*)+$'
        pattern = r'^(include\((.*?)\)\s*,\s*exclude\((.*?)\)\s*)+$' # include special characters in company names eg &.
        
        # Check if the input string matches the pattern
        if re.match(pattern, input_string):
            return True
        elif input_string == 'N/A':
            return True
        else:
            return False


    # Validation for Targeted Employer
    try:
        # User/Employer Template input validation
        if not (validate_input_format(criteria_df['Input']['Targeted Employer'])): 
            return render_template('main.html', batch_token=batch_token, error_status=f"""Error: Invalid input format for 'Targeted Employer'. Input should follow the format: 'Example: True - "include(Shell, BP) ,  exclude( KLCC, Novella Clinical, Fidelity Investments), True - "include(), exclude()", True - "include(Shell, BP) , exclude()", True - "include() , exclude(Shell, BP)", False -  "include() , exclude(Shell," """)
    except Exception as e:
        return render_template('main.html', batch_token=batch_token, error_status=f"""Error: Invalid input format for 'Targeted Employer'. Input should follow the format: 'Example: True - "include(Shell, BP) ,  exclude( KLCC, Novella Clinical, Fidelity Investments), True - "include(), exclude()", True - "include(Shell, BP) , exclude()", True - "include() , exclude(Shell, BP)", False -  "include() , exclude(Shell," """)
    
    
    # Validation for Age
    try:
        input_value = criteria_df['Input']['Age']
        if input_value == 'N/A':
            input_value = "0"
        # Define the expected format patterns
        expected_format_patterns = [
            r'\d+$',  # Option 1: Number
            r'\d+-\d+$',  # Option 2: Number-Number
            r'>\d+$',  # Option 3: >Number
            r'<\d+$'  # Option 4: <Number
        ]
        # Check if the input matches any of the expected format patterns
        if not any(re.match(pattern, input_value) for pattern in expected_format_patterns):
            return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Age'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Age'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")



    # Validation for Language
    try:
        values = [value.strip() for value in criteria_df['Input']['Language'].split(',')]
        if values:
            print(values)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Language'. Input should follow the format: '[Value], [Value], [Value]'")

    # Validation for Expected Salary
    try:
        input_value = criteria_df['Input']['Expected Salary in RM']
        if input_value == 'N/A':
            input_value = "0"
        # Define the expected format patterns
        expected_format_patterns = [
            r'\d+$',  # Option 1: Number
            r'\d+-\d+$',  # Option 2: Number-Number
            r'>\d+$',  # Option 3: >Number
            r'<\d+$'  # Option 4: <Number
        ]
        # Check if the input matches any of the expected format patterns
        if not any(re.match(pattern, input_value) for pattern in expected_format_patterns):
            return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Expected Salary in RM'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Invalid input format for 'Expected Salary in RM'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")

    # Validation for Years of Graduation
    try:
        float_value = criteria_df['Input']['Years of Graduation']
        if float_value == 'N/A':
            float_value = "0"
        float_value = int(float_value)
    except:
        return render_template('main.html', batch_token=batch_token, error_status=f"Error: Please make sure you only input the exact Year of Graduation. Example: '2024' ")

from enum import Enum, unique
import os
@unique
class EmailType(Enum):
    EMAIL_TYPE_PARSE_STARTED = "Email Type: Resume parser has started"
    EMAIL_TYPE_PARSE_COMPLETED = "Email Type: Resume parser has ended"
    EMAIL_TYPE_PARSE_ERROR = "Email Type: Error encountered "


def send_email(email_type, email_subject, email_body, recipient_email, attachment_path=None, attachment_filename=None):
    """
    Sends an email to the specified recipient with an optional attachment.

    Parameters:
    email_type: type of event triggering the email  
    email_subject (str): The subject of the email.
    email_body (str): The body text of the email.
    recipient_email (str): The recipient's email address.
    attachment_path (str, optional): The file system path to the attachment. Defaults to None.
    attachment_filename (str, optional): The filename for the attachment as it should appear in the email. Defaults to None.
    """
    try:
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        sender_email = 'yjyejui626@gmail.com'
        sender_password = os.getenv('GMAIL_SMTP_CVPARSER') # Ensure to use a secure method to store and retrieve passwords
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = email_subject
        msg.attach(MIMEText(email_body, 'plain'))

        if attachment_path and attachment_filename:
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= {attachment_filename}")
                msg.attach(part)

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Email sent successfully! Type: {email_type}")
    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")