import boto3
import difflib

bucket = ''
photo = ''

client = boto3.client('rekognition')

response = client.detect_text(Image={
    'S3Object': {
        'Bucket': 'wow-classifier',
        'Name': 'freehold6.png'
    }
})

text_detections = response['TextDetections']
print(response)
for text in text_detections:
    print('Detected text: {}'.format(text['DetectedText']))
    print('Confidence: {:.2f}%'.format(text['Confidence']))
    print('Id: {}'.format(text['Id']))
    if 'ParentId' in text:
        print('Parent Id: {}'.format(text['ParentId']))
    print('Type: {}'.format(text['Type']))
    print()