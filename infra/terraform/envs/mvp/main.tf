data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

locals {
  vpc_id = var.vpc_id == "" ? data.aws_vpc.default[0].id : var.vpc_id
}

data "aws_subnets" "default" {
  count = var.subnet_id == "" ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }
}

data "aws_ami" "ubuntu" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  subnet_id = var.subnet_id == "" ? tolist(data.aws_subnets.default[0].ids)[0] : var.subnet_id
  ami_id    = var.ami_id == "" ? data.aws_ami.ubuntu[0].id : var.ami_id

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_iam_role" "app" {
  name = "${var.project_name}-${var.environment}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.project_name}-${var.environment}-app-profile"
  role = aws_iam_role.app.name
}

resource "aws_security_group" "app" {
  name_prefix = "${var.project_name}-${var.environment}-app-"
  description = "llmenhance app EC2 security group"
  vpc_id      = local.vpc_id

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.environment}-app"
  })
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.app.id
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_ingress_rule" "streamlit" {
  for_each = var.streamlit_allowed_cidr_blocks

  security_group_id = aws_security_group.app.id
  description       = "Streamlit frontend"
  ip_protocol       = "tcp"
  from_port         = 8501
  to_port           = 8501
  cidr_ipv4         = each.value
}

resource "aws_vpc_security_group_ingress_rule" "model_ollama_from_app" {
  count = var.model_server_security_group_id == "" ? 0 : 1

  security_group_id            = var.model_server_security_group_id
  referenced_security_group_id = aws_security_group.app.id
  description                  = "Ollama from llmenhance app EC2"
  ip_protocol                  = "tcp"
  from_port                    = var.model_server_ollama_port
  to_port                      = var.model_server_ollama_port
}

resource "aws_instance" "app" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.app.name
  associate_public_ip_address = true

  user_data = templatefile("${path.module}/user_data_app.sh.tftpl", {
    repo_url        = var.repo_url
    repo_ref        = var.repo_ref
    ollama_base_url = var.ollama_base_url
    llm_model       = var.llm_model
    embedding_model = var.embedding_model
  })
  user_data_replace_on_change = true

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    tags = merge(local.tags, {
      Name   = "${var.project_name}-${var.environment}-app-root"
      Backup = "${var.project_name}-${var.environment}"
    })
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.environment}-app"
  })
}

resource "aws_eip" "app" {
  domain   = "vpc"
  instance = aws_instance.app.id

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.environment}-app-eip"
  })
}
