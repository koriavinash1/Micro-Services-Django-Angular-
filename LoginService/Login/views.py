# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from rest_framework.decorators import api_view

# Create your views here.

def login_loginpage(request):
	# view function for the product page
	return render(request, 'index.html',{})