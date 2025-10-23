# 🚀 배포 가이드

AI Paper Newsletter 시스템을 프로덕션 환경에 배포하는 방법을 설명합니다.

## 📋 목차

1. [AWS EC2 배포](#aws-ec2-배포)
2. [Docker 프로덕션 설정](#docker-프로덕션-설정)
3. [보안 설정](#보안-설정)
4. [모니터링 설정](#모니터링-설정)
5. [백업 및 복구](#백업-및-복구)

## AWS EC2 배포

### 1. EC2 인스턴스 설정

#### 권장 사양
- **인스턴스 타입**: t3.medium 이상 (2 vCPU, 4GB RAM)
- **스토리지**: 30GB 이상 (SSD)
- **OS**: Ubuntu 24.04 LTS

#### 인스턴스 생성

```bash
# AWS CLI로 인스턴스 생성
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx \
  --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3} \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-paper-newsletter}]'
```

### 2. 보안 그룹 설정

```bash
# HTTP/HTTPS 허용 (Load Balancer 사용 시)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# 내부 서비스 포트 (VPC 내에서만 접근)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 8000 \
  --source-group sg-xxxxx

# n8n 포트 (VPN을 통해서만 접근)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 5678 \
  --cidr 10.0.0.0/16
```

### 3. IAM 역할 설정

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-paper-bucket",
        "arn:aws:s3:::your-paper-bucket/*"
      ]
    }
  ]
}
```

EC2 인스턴스에 IAM 역할 연결:

```bash
aws ec2 associate-iam-instance-profile \
  --instance-id i-xxxxx \
  --iam-instance-profile Name=PaperProcessorRole
```

### 4. 서버 초기 설정

```bash
# SSH 접속
ssh -i your-key.pem ubuntu@ec2-xxx-xxx-xxx-xxx.compute.amazonaws.com

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Git 설치 및 프로젝트 클론
sudo apt install -y git
git clone <your-repo-url>
cd ai-paper-newsletter
```

### 5. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env
nano .env

# AWS Secrets Manager 사용 (권장)
aws secretsmanager create-secret \
  --name ai-paper-newsletter/prod \
  --secret-string file://.env

# 시크릿 가져오기
aws secretsmanager get-secret-value \
  --secret-id ai-paper-newsletter/prod \
  --query SecretString \
  --output text > .env
```

### 6. 배포 실행

```bash
# Docker Compose로 실행
docker-compose -f docker-compose.prod.yml up -d

# 로그 확인
docker-compose logs -f
```

## Docker 프로덕션 설정

### docker-compose.prod.yml

```yaml
version: '3.8'

services:
  paper-processor:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ai-paper-processor
    restart: always
    ports:
      - "8000:8000"
    environment:
      - ENV=production
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock:ro
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  n8n:
    image: n8nio/n8n:latest
    container_name: n8n
    restart: always
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_HOST=0.0.0.0
      - N8N_PORT=5678
      - N8N_PROTOCOL=https
      - NODE_ENV=production
      - WEBHOOK_URL=https://your-domain.com/
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      paper-processor:
        condition: service_healthy
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: nginx
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - paper-processor
      - n8n

volumes:
  n8n_data:
    driver: local

networks:
  default:
    name: ai-paper-network
```

### Nginx 설정

`nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream paper_processor {
        server paper-processor:8000;
    }

    upstream n8n {
        server n8n:5678;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    # Paper Processor API
    server {
        listen 443 ssl http2;
        server_name api.your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://paper_processor;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

    # n8n
    server {
        listen 443 ssl http2;
        server_name n8n.your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://n8n;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## 보안 설정

### 1. SSL/TLS 인증서

```bash
# Let's Encrypt 사용
sudo apt install certbot python3-certbot-nginx

# 인증서 발급
sudo certbot --nginx -d api.your-domain.com -d n8n.your-domain.com

# 자동 갱신 설정
sudo crontab -e
# 추가: 0 0 * * 0 certbot renew --quiet
```

### 2. 방화벽 설정

```bash
# UFW 설정
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable
```

### 3. 환경 변수 암호화

```bash
# AWS Secrets Manager 사용
# 또는 Vault 사용
docker run --cap-add=IPC_LOCK -e 'VAULT_DEV_ROOT_TOKEN_ID=myroot' vault
```

### 4. API Rate Limiting

Nginx에서 rate limiting 설정:

```nginx
http {
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    server {
        location /process {
            limit_req zone=api_limit burst=20;
            proxy_pass http://paper_processor;
        }
    }
}
```

## 모니터링 설정

### 1. CloudWatch 로그 전송

```bash
# CloudWatch Logs Agent 설치
sudo apt install -y amazon-cloudwatch-agent

# 설정 파일
sudo nano /opt/aws/amazon-cloudwatch-agent/etc/config.json
```

```json
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/home/ubuntu/ai-paper-newsletter/logs/app.log",
            "log_group_name": "/aws/ec2/ai-paper-newsletter",
            "log_stream_name": "{instance_id}/application"
          }
        ]
      }
    }
  }
}
```

### 2. 헬스 체크 모니터링

```python
# healthcheck.py
import requests
import boto3
from datetime import datetime

def check_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def send_alert(message):
    sns = boto3.client('sns')
    sns.publish(
        TopicArn='arn:aws:sns:region:account:alerts',
        Subject='AI Paper Newsletter Alert',
        Message=message
    )

if not check_health():
    send_alert(f"Health check failed at {datetime.now()}")
```

### 3. Grafana 대시보드

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  grafana_data:
```

## 백업 및 복구

### 1. 자동 백업 스크립트

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup"

# n8n 데이터 백업
docker exec n8n n8n export:workflow --all --output=/tmp/workflows.json
docker cp n8n:/tmp/workflows.json $BACKUP_DIR/n8n_workflows_$DATE.json

# 로그 백업
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# S3로 업로드
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/ai-paper-newsletter/

# 30일 이상 된 백업 삭제
find $BACKUP_DIR -type f -mtime +30 -delete
```

### 2. Cron 설정

```bash
# 매일 새벽 2시에 백업
crontab -e
# 추가: 0 2 * * * /home/ubuntu/ai-paper-newsletter/backup.sh
```

### 3. 복구 절차

```bash
# n8n 워크플로우 복구
docker cp backup/n8n_workflows_YYYYMMDD.json n8n:/tmp/workflows.json
docker exec n8n n8n import:workflow --input=/tmp/workflows.json

# 로그 복구
tar -xzf backup/logs_YYYYMMDD.tar.gz
```

## 성능 최적화

### 1. Docker 리소스 제한

```yaml
services:
  paper-processor:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 2. Redis 캐싱 (선택사항)

```yaml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
```

### 3. 로그 로테이션

```bash
# /etc/logrotate.d/ai-paper-newsletter
/home/ubuntu/ai-paper-newsletter/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

## 트러블슈팅

### 서비스가 시작되지 않을 때

```bash
# 로그 확인
docker-compose logs paper-processor

# 컨테이너 상태 확인
docker ps -a

# 리소스 사용량 확인
docker stats
```

### 메모리 부족

```bash
# Swap 메모리 추가
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 포트 충돌

```bash
# 포트 사용 확인
sudo netstat -tulpn | grep :8000

# 프로세스 종료
sudo kill -9 <PID>
```

## 유지보수

### 정기 업데이트

```bash
# 1. 백업
./backup.sh

# 2. 코드 업데이트
git pull origin main

# 3. 컨테이너 재시작
docker-compose -f docker-compose.prod.yml up -d --build

# 4. 헬스 체크
curl http://localhost:8000/health
```

### 로그 모니터링

```bash
# 실시간 로그
docker-compose logs -f

# 에러만 필터링
docker-compose logs | grep ERROR

# 특정 시간 범위
docker-compose logs --since="2024-01-01" --until="2024-01-31"
```

---

**배포 체크리스트:**

- [ ] EC2 인스턴스 생성 및 설정
- [ ] 보안 그룹 설정
- [ ] IAM 역할 연결
- [ ] Docker 및 Docker Compose 설치
- [ ] 환경 변수 설정
- [ ] SSL 인증서 설정
- [ ] 방화벽 설정
- [ ] 모니터링 설정
- [ ] 백업 스크립트 설정
- [ ] 헬스 체크 테스트
- [ ] n8n 워크플로우 import
- [ ] 프로덕션 테스트