# program to upload file to s3 bucket
import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def upload_file(
    file_name: Path,
    bucket: str,
    object_name: str = None,
    extra_id: dict = {"Tagging": None},
):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :param  extra_id : dict = {'Tagging': None}
    :return: True if file was uploaded, else False
    """
    session = boto3.Session(
        aws_access_key_id=os.environ["AWSAccessKeyId"],
        aws_secret_access_key=os.environ["AWSSecretKey"],
    )
    s3 = session.resource("s3")
    s3_client = s3.Bucket(bucket)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    try:
        response = s3_client.upload_file(file_name, object_name, ExtraArgs=extra_id)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """
    session = boto3.Session(
        aws_access_key_id=os.environ["AWSAccessKeyId"],
        aws_secret_access_key=os.environ["AWSSecretKey"],
    )
    s3_client = session.client("s3")

    # Create bucket
    try:
        if region is None:
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client("s3", region_name=region)
            location = {"LocationConstraint": region}
            s3_client.create_bucket(
                Bucket=bucket_name, CreateBucketConfiguration=location
            )
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_file(
    file_name: Path,
    bucket: str = "projectbirdclassifier" ,
    object_name: str = None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :param  extra_id : dict = {'Tagging': None}
    :return: True if file was uploaded, else False
    """
    session = boto3.Session(
        aws_access_key_id=os.environ["AWSAccessKeyId"],
        aws_secret_access_key=os.environ["AWSSecretKey"],
    )
    s3 = session.resource("s3")
    s3_client = s3.Bucket(bucket)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    try:
        response = s3_client.upload_file(file_name, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def s3_download_model(path: str, bucket_name, key_name: str):
    """ ""

    Args:
        path (str): full path with the filename eg: Bird_species_classification/configs/config.yaml
        bucket_name (_type_): name of the key you want to store
        key_name (str): key of the file

    Returns:
        True if file was Download, else False
    """
    try:
        session = boto3.Session(
            aws_access_key_id=os.environ["AWSAccessKeyId"],
            aws_secret_access_key=os.environ["AWSSecretKey"],
        )
        # Creating S3 Resource From the Session.
        s3 = session.resource("s3")
        bucket = s3.Bucket(bucket_name)
        obj = bucket.objects.filter()
        file_key = [i for i in obj if key_name in i.key][0]
        bucket.download_file(file_key.key, path)  # save to same path
        logging.info("Downloaded Model From S3")

    except ClientError as e:
        logging.error(e)
    except IndexError as e:
        logging.error(e)
        return False
    return True

def get_best_model_s3(best_model_path , key = "Best_model" ,  bucket_name="projectbirdclassifier" ):
        key = "Best_model"
        with open(best_model_path , "w") as file_path :
                status= s3_download_model(path =str(best_model_path) , bucket_name=bucket_name ,key_name=key )
                if not status:
                    best_model_path= None
        
        return best_model_path
