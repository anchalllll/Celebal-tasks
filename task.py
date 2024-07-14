import datetime
import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient

def main(mytimer: func.TimerRequest) -> None:
    today = datetime.datetime.utcnow().date()
    last_saturday = today - datetime.timedelta(days=today.weekday() + 2)
    if today == last_saturday and today.day > 24:
        # Trigger ADF pipeline
        subscription_id = 'your-subscription-id'
        resource_group_name = 'your-resource-group-name'
        factory_name = 'your-data-factory-name'
        pipeline_name = 'your-pipeline-name'

        credential = DefaultAzureCredential()
        adf_client = DataFactoryManagementClient(credential, subscription_id)
        
        run_response = adf_client.pipelines.create_run(resource_group_name, factory_name, pipeline_name)
        print(f"Pipeline run ID: {run_response.run_id}")

