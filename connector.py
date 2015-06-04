import requests
import json
import datetime

r = requests.get("https://webservices.iso-ne.com/api/v1.1/fiveminutelmp/current.json",auth=('alexander.s.dagley@gmail.com','golfer01'))
a=r.json()
lmp= a['FiveMinLmps'].values()[0][0]['LmpTotal']
c = requests.get("http://api.openweathermap.org/data/2.5/weather?q=Boston,us&units=imperial")
c= c.json()
tmp= c["main"]["temp"]
ct=str(datetime.datetime.now())
ct=ct.split(' ')
ct=ct[1].split(':')
ct=str(int(ct[0])*60+int(ct[1]))

print "lmp: ", lmp
print "tmp: ", tmp
print "time: ", ct

