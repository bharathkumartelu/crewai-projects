from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, GithubSearchTool, ScrapeWebsiteTool
import os
from dotenv import load_dotenv

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class DocumentationSlackBot():
	"""DocumentationSlackBot crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	load_dotenv()

	docs_tool = WebsiteSearchTool(website='https://www.nextflow.io/docs/latest/index.html')
	scrape_tool = ScrapeWebsiteTool()
	github_search = GithubSearchTool(
		gh_token=os.getenv("GITHUB_TOKEN"),
		content_types=['code', 'repo'] # Options: code, repo, pr, issue
	)

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	# @agent
	# def bioinformatics_expert(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['bioinformatics_expert'],
	# 		verbose=True,
	# 		tools=[self.scrape_tool]
	# 	)

	@agent
	def github_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['github_analyst'],
			tools=[self.github_search],
			allow_delegation=True
		)
	
	@agent
	def documentation_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['documentation_specialist'],
			tools=[self.docs_tool, self.scrape_tool],
			allow_delegation=True
		)

	@agent
	def google_cloud_batch_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['google_cloud_batch_expert'],
			tools=[self.scrape_tool],
			allow_delegation=True
		)
	
	@agent
	def answer_specialist(self) -> Agent:
		return Agent(
			config=self.agents_config['answer_specialist'],
			allow_delegation=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	# @task
	# def bio_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['bio_task'],
	# 		agent=self.bioinformatics_expert(),
	# 	)
	
	@task
	def github_task(self) -> Task:
		return Task(
			config=self.tasks_config['github_task'],
			agent=self.github_analyst(),
		)
	
	@task
	def doc_task(self) -> Task:
		return Task(
			config=self.tasks_config['doc_task'],
			agent=self.documentation_specialist(),
		)

	@task
	def gcloud_task(self) -> Task:
		return Task(
			config=self.tasks_config['gcloud_task'],
			agent=self.google_cloud_batch_expert(),
		)
	
	@task
	def answer_task(self) -> Task:
		return Task(
			config=self.tasks_config['detailed_answer'],
			output_file='answer.txt',
			agent=self.answer_specialist(),
			context=[self.gcloud_task(), self.doc_task(), self.github_task()]
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the DocumentationSlackBot crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			# manager_llm="gpt-4o"
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

