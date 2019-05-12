import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_mail(sender, to, score , timestamp):

	sender_email_address = sender
	sender_email_password = 'sctce123'
	receiver_email_address = to

	email_subject_line = 'Security Alert from Vision.ai'

	msg = MIMEMultipart()
	msg['From'] = sender_email_address
	msg['To'] = receiver_email_address
	msg['Subject'] = email_subject_line

	email_body = 'Dear User,\n \
	 							\nThis is a security alert for your surveillance camera.\
								\nAn anomaly has been detected. Check activity.\
								\n'+str("Time of occurence: ")+\
								str(timestamp)+\
								str("\nAnomaly score: ")+\
								str(score)+str("\n \nThanks,\nTeam Vision")
	msg.attach(MIMEText(email_body, 'plain'))

	email_content = msg.as_string()
	try:
		server = smtplib.SMTP('smtp.gmail.com:587')
		server.starttls()
		server.login(sender_email_address, sender_email_password)

		server.sendmail(sender_email_address, receiver_email_address, email_content)
		server.quit()
	except Exception as E:
		print("e-mail failed, network currently unavailable.")

if __name__ == '__main__':
	send_mail('vision.ai.updates@gmail.com', 'nevinbaiju@gmail.com', 0.5, '2019-03-29 11:40:12')
