import smtplib
from random import SystemRandom
import string

mf = open('mail.txt', 'rt')
cfg = open('mail.cfg', 'rt')

(USER, PASS) = map(string.strip, cfg.readlines())

lines = map(string.strip, mf.readlines())

random = SystemRandom()

rebalance = random.choice(range(30)) == 0

server = smtplib.SMTP('smtp.gmail.com:587')
server.ehlo()
server.starttls()

msg = "\r\n".join(["From: %s" % USER, "To: %s" % USER, "Subject: Stocks %s" % rebalance,
    "",
    ] + lines)

server.login(USER,PASS)

server.sendmail(USER, USER, msg)

server.quit()
