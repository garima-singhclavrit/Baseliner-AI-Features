# import serializer from rest_framework
from rest_framework import serializers
	
# import model from models.py
from .models import new_model

# Create a model serializer
class NewSerializer(serializers.ModelSerializer):
	# specify model and fields
	class Meta:
		model = new_model
		fields = '__all__'
# import serializer from rest_framework
# from rest_framework import serializers

# # import model from models.py
# from .models import new_model, task_estimation_result

# # Create a model serializer
# class NewSerializer(serializers.ModelSerializer):
# 	class Meta:
# 		model = new_model
# 		fields = [
# 			'task_id',
# 			'project_id',
# 			'project_name',
# 			'task_name',
# 			'task_descrption',
# 			'task_label',
# 			'planned_estimate',
# 			'sprint_number',
# 			'priority',
# 			'Optimistic_estimate',
# 			'Most_likely_estimate',
# 			'Pessimistic_estimate',
# 			'actual_estimate',
# 		]


# class EstimationResultSerializer(serializers.ModelSerializer):
# 	class Meta:
# 		model = task_estimation_result
# 		fields = '__all__'
