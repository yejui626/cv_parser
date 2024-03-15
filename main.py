import os
import asyncio
import logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import re
from utils import parse_criteria_excel,send_email, EmailType
from datetime import datetime, timedelta
from Resume_openai_promt import ResumeParser
import pandas as pd
import zipfile
from flask import Flask, request, render_template, send_file, jsonify, make_response
import secrets
import string
import threading
import openai
from dotenv import load_dotenv
import openpyxl
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = "upload"
DOWNLOAD_FOLDER = "download"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER

# Load environment variables8
load_dotenv("./conf/.env")
#load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')


status_dict = {}
failed_resumes_info = []

@app.route('/', methods=['GET', 'POST'])
def index():
    batch_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))

    return render_template('main.html', batch_token=batch_token)


@app.route('/download', methods=['POST'])
def download_file():
    requested_file_name = request.form.get('file_name')

    if not requested_file_name:
        return jsonify({'error': 'Please provide a file name.'}), 400

    output_dir = os.path.join(os.getcwd(), 'download/')
    #output_dir = os.path.join(os.getcwd(), 'upload/')
    file_path = os.path.join(output_dir, requested_file_name)

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found.'}), 404

    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': 'Error while downloading the file.'}), 500
    
@app.route('/download_criteria', methods=['GET'])
def download_criteria():
    criteria_file_path = "Criteria_cv_matching.xlsx"  # Update with the actual path to your file

    if not os.path.isfile(criteria_file_path):
        return jsonify({'error': 'File not found.'}), 404
    try:
        return send_file(criteria_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': 'Error while downloading the file.'}), 500
    
@app.route('/upload_files', methods=['POST'])
def upload_files_resumes():
    print("this function is run now", request.form)
    try:
        print("printing initia status_dict",status_dict)                
        current_chunk = int(request.form['dzchunkindex'])
        submitted_token = request.form.get('batch_token', None)
        upload_dir = os.path.join(os.getcwd(), f"upload/")
        pdf_dir = os.path.join(os.getcwd(), f"upload/{submitted_token}")
        output_dir = os.path.join(os.getcwd(), 'download/')
        uploaded_file = request.files.get('file')
        
        if submitted_token not in status_dict:
            status_dict[submitted_token] = {'total_resumes_uploaded':0,'resumes_upload_succeeded_count':0,'resumes_upload_failed_count':0}
            
        if 'total_resumes_uploaded' in status_dict[submitted_token] and current_chunk == 0:
            print("printing total before adding ",status_dict[submitted_token]['total_resumes_uploaded'])
            status_dict[submitted_token]['total_resumes_uploaded'] += 1
        
        if not os.path.exists(upload_dir):
            os.mkdir(upload_dir)
            
        if not os.path.exists(pdf_dir):
            os.mkdir(pdf_dir)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        
        
        tmp_file_path = os.path.join(pdf_dir, uploaded_file.filename)
        
        
        try:
            with open(tmp_file_path, 'ab') as f:
                f.seek(int(request.form['dzchunkbyteoffset']))
                f.write(uploaded_file.stream.read())
                if 'resumes_upload_succeeded_count' in status_dict[submitted_token] and current_chunk == 0:
                    status_dict[submitted_token]['resumes_upload_succeeded_count'] += 1
        except Exception as e:
            if 'resumes_upload_failed_count' in status_dict[submitted_token] and current_chunk == 0:
                status_dict[submitted_token]['resumes_upload_failed_count'] += 1
            failed_resumes_info.append({
                'filename': uploaded_file.filename,
                'reason': e
            })
            failed_resumes_info.append(uploaded_file)   
            print (failed_resumes_info)

            return make_response(("Not sure why,"
                                " but we couldn't write the file to disk", 500))
            
        total_chunks = int(request.form['dztotalchunkcount'])
        
        if current_chunk + 1 == total_chunks:
            print(f'File {uploaded_file.filename} has been uploaded successfully')
        else:    
            print(f'Chunk {current_chunk + 1} of {total_chunks} '
                    f'for file {uploaded_file.filename} complete')
            
        print(status_dict)
        print("printing token",submitted_token)                 
        print("printing total",status_dict[submitted_token]['total_resumes_uploaded'])
        print("printing succeed",status_dict[submitted_token]['resumes_upload_succeeded_count'])
        
        return jsonify({'status': 'done uploaded'})
    except Exception as e:
        if 'resumes_upload_failed_count' in status_dict[submitted_token]:
            status_dict[submitted_token]['resumes_upload_failed_count'] += 1
        failed_resumes_info.append({
            'filename': uploaded_file.filename,
            'reason': jsonify({'status': e})
        }) 
        print ("printing",failed_resumes_info)

        return jsonify({'status': e})


def save_to_txt(text, filename, encoding='utf-8'):
    '''
    Helper function to save text into file
    '''
    with open(filename, 'w', encoding=encoding) as file:
        file.write(text)
    print("Text saved to", filename)

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Upload and process files, including resumes, job-related information, and applicant categories.

    Returns:
    Response: A response indicating the start of file processing in the background.
    """
    try:
        submitted_token = request.form.get('batch_token', None)
        current_date, current_datetime = str(datetime.today())[0:10], re.sub(' ', '_', re.sub(':', '', str(datetime.today())[0:19]))
        result_file_name= f"ResumeParserOutput{current_datetime}_{submitted_token}.zip"
        start_time = datetime.now()
        pdf_dir = os.path.join(os.getcwd(), f"upload/{submitted_token}")
        output_dir = os.path.join(os.getcwd(), 'download/') # output folder

        # If upload folder does not exist, create a new upload folder
        if not os.path.exists(pdf_dir):
            os.mkdir(pdf_dir)

        # If output folder does not exist, create a new output folder
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Retrieve the files uploaded through the webapp
        FILENAMES = []
        files = request.files.getlist('file')
        app.logger.info(files)
        
        # Get Job Title
        job_title = request.form['job_title']
        print (f'Getting job_title {job_title.capitalize()}')

        # Saving resumes 
        # Initialising resume upload statuses      
        approx_time_per_resume = 30 # In seconds
        approx_buffer = 10  # Additional buffer time in seconds for retries and sleep

        for file in files:
            file_name = file.filename
            filename_parts = file_name.rsplit('.', 1)
            if len(filename_parts) == 2: # if the filename has at least one dot
                name_without_dots = filename_parts[0].replace('.', '') # remove all dots from name before last dot
                file_name = name_without_dots + '.' + filename_parts[1] # join the name parts without dots and the extension
            else: # if the filename doesn't have any dots
                file_name = file_name
            FILENAMES.append(file_name)

            # Save the resume files
            if file_name.upper().endswith('.PDF') or file_name.upper().endswith('.DOC') or file_name.upper().endswith('.DOCX'):
                print (f"File uploaded: {file_name}\t total_resumes_uploaded: {status_dict[submitted_token]['total_resumes_uploaded']}")
                try:
                    path = os.path.join(pdf_dir, file_name) # Save the pdf file into Upload folder
                    file.save(path)
                except Exception as e:

                    return str(e)

        # Processing criteria file            
        file_criteria = request.files.getlist('file_criteria')
        criteria_file_name = ""
        # Save the criteria_info file
        allowed_extensions = ['.xlsx']
        criteria_info = ''
        for file in file_criteria:
            filename = file.filename
            extension = os.path.splitext(filename)[1]
            if extension in allowed_extensions:
                criteria_info = file.filename
                if criteria_info is not '':
                    try:
                        splitx = criteria_info.split('.')[-1]
                        criteria_file_name = filename
                        criteria_info_sav = os.path.join(pdf_dir, f'{filename}')
                        file.save(criteria_info_sav)
                        file.close()
                    except Exception as e:
                        criteria_info_sav = ''
                        break
        
        
        # Get Job Description
        job_description = request.form['job-description']

        # Limit to 1200 characters.
        try:
            if len(job_description) >= 3500:
                # Remove line spaces
                cleaned_string = re.sub(r'\n+', '\n', job_description)
                job_description = cleaned_string[:3500]
            save_to_txt(job_description, os.path.join(pdf_dir, 'Job Description.txt'))
        except Exception as e:
            return str(e)

        # Get Job Requirement
        job_requirement = request.form['job-requirement']
        # Limit to 1200 characters.
        try:
            if len(job_requirement) >= 3500:
                # Remove line spaces
                cleaned_string = re.sub(r'\n+', '\n', job_requirement)
                job_requirement = cleaned_string[:3500]
            save_to_txt(job_requirement, os.path.join(pdf_dir, 'Job requirement.txt'))
        except Exception as e:
            return str(e)

        applicant_category = request.form.get('applicant-category')
        df_category=pd.DataFrame()
        # Append new data
        for file in files:
            file_name = file.filename
            index = file_name.rsplit('.', 1)[0].replace('.', '')
            df_category.loc[index, 'applicant_category'] = applicant_category
        # Save the data to Excel
        df_category.to_excel(os.path.join(pdf_dir, 'category.xlsx'))
        
        recipient_email = request.form.get('email')

        # Current time as the program start time for demonstration
        program_start_time = datetime.now()+timedelta(hours=8)

        # Calculations for program performance
        approx_total_program_time = status_dict[submitted_token]['resumes_upload_succeeded_count'] * approx_time_per_resume + approx_buffer
        program_end_time = program_start_time + timedelta(seconds=approx_total_program_time)
        
        prog_hours, remainder_minutes = divmod(approx_total_program_time, 3600)
        prog_minutes, prog_seconds = divmod(remainder_minutes, 60)
        approx_total_in_hmd = f"{prog_hours} hours {prog_minutes} minutes {prog_seconds} seconds"

        # Program Performance Information
        program_performance_info = {
            "approx_time_per_resume": approx_time_per_resume,
            "approx_buffer": approx_buffer,
            "approx_total_program_time": approx_total_program_time,
            "start_time": program_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": program_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "actual_time_taken": 0 # assigned after program ends for  perfromance evaluation
        }

        # Create the resume_status dictionary incorporating program performance
        resume_status = {
            "total_resumes_uploaded": status_dict[submitted_token]['total_resumes_uploaded'],
            "resumes_upload_succeeded_count": status_dict[submitted_token]['resumes_upload_succeeded_count'],
            "resumes_upload_failed_count": status_dict[submitted_token]['resumes_upload_failed_count'],
            "failed_resumes_info":failed_resumes_info,
            "approx_total_program_time": approx_total_program_time,
            "approx_total_in_hmd": approx_total_in_hmd,
            "program_start_time": program_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "program_end_time_estimation": program_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "program_performance_info": program_performance_info,  # Embed the performance info dictionary
        }
        
        batch_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        
        #### PARSING OF THE CRITERIA EXCEL FILE ####
        # Open the Job Description file in read mode
        file_path = os.path.join(pdf_dir, criteria_file_name)
        
        response = parse_criteria_excel(file_path)

        if response is None:
            # Preprocessing of the criteria Excel file
            criteria_df = pd.read_excel(file_path,skiprows=4,header=1,skipfooter=1)
            ## List of specified column names
            specified_columns = ['Selected', 'Criterias', 'Priority', 'Weightage', 'Input']
            ## Drop columns that are not in the specified list
            criteria_df = criteria_df.drop(columns=criteria_df.columns.difference(specified_columns))
            ## Set "Criterias" column as the index
            criteria_df.set_index('Criterias', inplace=True)

            # Replace NaN values in 'Input' column with "N/A"
            criteria_df['Input'].fillna('N/A', inplace=True)
            # Parse 'Selected' column into True and False
            criteria_df['Selected'] = criteria_df['Selected'].apply(lambda x: True if '✔' in str(x) else False)
            
            background_thread = threading.Thread(target=background_task, args=(pdf_dir, resume_status, job_title, job_description, output_dir, current_datetime, result_file_name, start_time, recipient_email, submitted_token, criteria_file_name,criteria_df))
            background_thread.start()

            return render_template('main.html', batch_token=batch_token, result_file_name=result_file_name, resume_status=resume_status)
        else:
            return response
    except Exception as e:
        return jsonify(f"Please make sure you have uploaded the correct files.'",e)

def background_task(pdf_dir, resume_status, job_title, job_description, output_dir, current_datetime, result_file_name, start_time, email,submitted_token,criteria_file_name,criteria_df):
    print("============================== Starting RESUME PARSER PART 1 ==============================")
    # Calculate hours and minutes using divmod
    estimated_time_seconds = resume_status['approx_total_program_time']
    prog_hours, remainder_minutes = divmod(estimated_time_seconds, 3600)
    prog_minutes, prog_seconds = divmod(remainder_minutes, 60)
    
    
    # Send email to user to notify parser has started
    start_email_subject= f"'Resume Parsing Started'. "
    # start_email_body = f"Job Parser ID: {job_parser_id}\n Your resume parsing has started.\n Upon successful completion, the results will be sent to your email."
    
    # Other info to add: resumes_upload_failed_count: {resume_status['resumes_upload_failed_count']},
    start_email_body = f""" 

    Dear Hiring Manager,
    Please be informed that the CV matching tool has commenced processing your resumes. 


    Submitted token: {submitted_token}\n

    Below is a summary of your submission::
    Resume processing start time: {resume_status['program_start_time']},
    Total number of submitted resumes: {resume_status['total_resumes_uploaded']},
    Total number of resumes processed without issue: {resume_status['resumes_upload_succeeded_count']},
    Approximate processing time: {prog_hours} hours {prog_minutes} minutes {prog_seconds} seconds,
    Estimation program end time: {resume_status['program_end_time_estimation']} \n
    
    Upon the successful completion of this process, you will promptly receive an email containing the results. 
    Alternatively, your designated output filename is {result_file_name}, which you can use it to download the output from the CV matching tool UI, but only upon completion. 
    We trust that this tool will effectively assist you in identifying suitable candidates for the role.
    
    Thank you and Best Regards.
    
    Note: This is an auto-generated email. Please refrain from replying to this email.

    """
    # start_email_body +=  f" \nYour resume parsing has started.\nUpon successful completion, we will email you the results."
    send_email( EmailType.EMAIL_TYPE_PARSE_STARTED ,start_email_subject, start_email_body,  email)
    print (f"Email content:\n\t {start_email_body}")
    print(criteria_file_name)
    parser = ResumeParser(pdf_dir, job_title=job_title,job_description=job_description,criteria_file_name=criteria_file_name,criteria_df=criteria_df)
    parsed_resumes = asyncio.run(parser.parsing())
            
    # while running parsing(), send email if program stops to notify failure of program with attached error message, program run id to identitfy which run
    try:

        parsed_resume_filepath = os.path.join(output_dir, f'CVParser_{current_datetime}.xlsx')
        # Sort the DataFrame based on the average column
        parsed_resumes_sorted = parsed_resumes.sort_values(by='Total Overall Score', ascending=False)
        parsed_resumes_sorted.to_excel(parsed_resume_filepath, index=False)
        criteria_file_path = os.path.join(pdf_dir, criteria_file_name)
        print(criteria_file_path)

        # Code to zip the results and send email
        zip_path = os.path.join(output_dir, result_file_name)
        with zipfile.ZipFile(zip_path, mode='w') as archive:
            archive.write(parsed_resume_filepath, os.path.basename(parsed_resume_filepath))
            archive.write(criteria_file_path, os.path.basename(criteria_file_path))
        
        # EMAIL_TYPE_PARSE_COMPLETED contents 
        time_elapsed = datetime.now() - start_time
        total_program_runtime_str = f'Time elapsed (hh:mm:ss.ms) {format(time_elapsed)}'
        parse_completed_email_content = f"""Submitted token: {submitted_token}.
        
        Your resume parsing has been completed. Please find the results attached in the .zip file.
        Number of resumes parsed: {resume_status['resumes_upload_succeeded_count']}
        Totol Runtime: {total_program_runtime_str}
        """
        send_email(EmailType.EMAIL_TYPE_PARSE_COMPLETED,f'Resume Parsing Completed for submitted token: {submitted_token}',parse_completed_email_content , email, zip_path, result_file_name)
    except Exception as e:
        send_email(EmailType.EMAIL_TYPE_PARSE_ERROR,f'Resume Parsing Failed for submitted token: {submitted_token}', f'Submitted token: {submitted_token}.\n\nYour resume parsing failed with error: {str(e)}', email)
        print(f"Parsing failed. Job Parser ID:\t Error: {str(e)}\n Email sent to {email} to notify parsing failure.")
    
    try:
        shutil.rmtree(pdf_dir)
        print(f"Directory '{pdf_dir}' and its contents deleted successfully.")
    except Exception as e:
        print(f"Error deleting directory: {pdf_dir}")
        print(str(e))
            
    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
    

