#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from documentation_slack_bot.crew import DocumentationSlackBot

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        "question": "while running nextflow pipeline on gcp not all intermediate files were written to work directory only those specified in output directive for each process was written to work directory, but while running in local all intermediate files can be found in work directory how to change this behaviour of local to same as gcp"
    }
    
    try:
        DocumentationSlackBot().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


# def train():
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {
#         "question": "Define auto retry logic for VM preemption error exitcode 50001while using spot instance for nextflow run in google cloud"
#     }
#     try:
#         DocumentationSlackBot().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f"An error occurred while training the crew: {e}")

# def replay():
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         DocumentationSlackBot().crew().replay(task_id=sys.argv[1])

#     except Exception as e:
#         raise Exception(f"An error occurred while replaying the crew: {e}")

# def test():
#     """
#     Test the crew execution and returns the results.
#     """
#     inputs = {
#         'question': 'How to define auto retry exit codes for gcp job'
#     }
#     try:
#         DocumentationSlackBot().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

#     except Exception as e:
#         raise Exception(f"An error occurred while testing the crew: {e}")

