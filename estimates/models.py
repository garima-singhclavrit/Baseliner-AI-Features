from django.db import models

# Create your models here.
class new_model(models.Model):
    task_id = models.CharField(blank=True,max_length=200)
    task_name = models.CharField(blank=True,max_length=200)
    task_descrption = models.CharField(blank=True,max_length=200)
    task_label = models.CharField(blank=True,max_length=200)
    planned_estimate = models.CharField(blank=True,max_length=200)
    sprint_number = models.CharField(blank=True,max_length=200)
    priority = models.CharField(blank=True,max_length=200)
    Optimistic_estimate = models.CharField(blank=True,max_length=200)
    Most_likely_estimate = models.CharField(blank=True,max_length=200)
    Pessimistic_estimate = models.CharField(blank=True,max_length=200)
    # Actual_estimate = models.CharField(blank=True,max_length=200)


# class task_estimation_result(models.Model):
#     task = models.ForeignKey(new_model, on_delete=models.CASCADE, related_name='estimation_results')
#     ai_estimate = models.FloatField(null=True, blank=True)
#     three_point_estimate = models.FloatField(null=True, blank=True)
#     risk_factor = models.FloatField(null=True, blank=True)
#     is_replaced = models.CharField(blank=True, max_length=10)
#     created_at = models.DateTimeField(auto_now_add=True)

#     class Meta:
#         db_table = 'estimates_task_estimation_result'
