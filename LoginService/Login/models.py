# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.db import models

# Create your models here.
# TODO: add most followed company by user
#             company suggestion technique
#             user investment details
class User(models.Model) :
	GENDER_CHOICES = (
		('M','Male'),
		('F','Female'),
		('O','Other'),
	)
	USERTYPE_CHOICES = (
		(0,'Student'),
		(1,'Business Man'),
		(2,'Developer'),
		(3,'Admins'),
	)
	firstName = models.CharField(max_length=100)
	middleName = models.CharField(max_length=100,
					 default="",
					 blank=True)
	lastName = models.CharField(max_length=100,
					default="",
					blank=True)
	email = models.EmailField(unique=True)
	dob = models.DateField(null=True,
				    blank=True)
	password = models.CharField(max_length=100)
	gender = models.CharField(max_length=1,
					choices=GENDER_CHOICES,
					default='M')
	username = models.CharField(max_length=100,
					default="")
	phoneNumber = models.CharField(max_length=20,
						default="",
						blank=True)
	address = models.CharField(max_length=1000,
					default="",
					blank=True)
	userType = models.IntegerField(choices=USERTYPE_CHOICES)

	def getName(self):
		name = ""
		if self.firstName :
			name+= self.firstName
		if self.middleName :
			if name:
				name+= (" " + self.middleName)
			else :
				name = self.middleName
		if self.lastName:
			if name:
				name+= (" ") + self.lastName
			else:
				name = self.lastName
		return name

class OAuth(models.Model):
	pass