output "app_instance_id" {
  value = aws_instance.app.id
}

output "app_public_ip" {
  value = aws_eip.app.public_ip
}

output "streamlit_public_url" {
  value = "http://${aws_eip.app.public_ip}:8501"
}

output "ssm_streamlit_tunnel_command" {
  value = "aws ssm start-session --region ${var.aws_region} --target ${aws_instance.app.id} --document-name AWS-StartPortForwardingSession --parameters \"portNumber=8501,localPortNumber=8501\""
}

output "ssm_api_tunnel_command" {
  value = "aws ssm start-session --region ${var.aws_region} --target ${aws_instance.app.id} --document-name AWS-StartPortForwardingSession --parameters \"portNumber=8000,localPortNumber=8000\""
}
