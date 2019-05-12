"""
This script uses the sms api of way2sms.
More details on using the way2sms api
can be found here: https://gist.github.com/way2smscom
"""
import requests
import json

URL = 'https://www.way2sms.com/api/v1/sendCampaign'

# get request
def sendPostRequest(reqUrl, apiKey, secretKey, useType, phoneNo, senderId, textMessage):
	req_params = {
	'apikey':apiKey,
	'secret':secretKey,
	'usetype':useType,
	'phone': phoneNo,
	'message':textMessage,
	'senderid':senderId
	}
	return requests.post(reqUrl, req_params)

# get response
def send_sms(to_mobile, timestamp):
	provided_api_key = ''
	provided_secret = ''
	prod_stage = 'stage'
	active_sender_id = ''
	message_text = "An anomaly has occured under Cam 1 at {}, please attend to it. \n -vision.ai".format(timestamp)

	response = sendPostRequest(URL, provided_api_key, provided_secret, prod_stage, to_mobile, active_sender_id, message_text)
	if(response.text['status'] == 'success'):
		print("SMS successfully send.")

if __name__ == '__main__':
	send_sms("", "1:00 PM 12th May.")