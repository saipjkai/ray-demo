import boto3


def main():
    ec2 = boto3.resource('ec2', aws_access_key_id="aws_access_key_id", aws_secret_access_key="aws_secret_access_key")
    for instance in ec2.instances.all():
        print(
            "Id: {0}\nPlatform: {1}\nType: {2}\nPublic IPv4: {3}\nAMI: {4}\nState: {5}\n".format(
            instance.id, instance.platform, instance.instance_type, instance.public_ip_address, instance.image.id, instance.state
            )
        )


if __name__ == "__main__":
    main()
    