# Deploying the bookshop

## Requirements

- Python 3.12
- A PostgreSQL database
- An SMTP relay for customer emails

## Production rollout

1. Build the container image: `docker build -t bookshop:latest .`
2. Run database migrations: `bookshop migrate --database-url $DATABASE_URL`
3. Start the app behind the load balancer with `gunicorn bookshop.wsgi -w 4`.
4. Check `/healthz` returns 200 before shifting traffic.

## Rolling back

Redeploy the previous image tag. Migrations are backwards compatible for one release.
