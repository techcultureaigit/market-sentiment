# Manage the local portable PostgreSQL 16 cluster (no admin / no Windows service).
# Usage: .\scripts\pg.ps1 start | stop | status | psql | pgadmin
param([ValidateSet("start", "stop", "status", "psql", "pgadmin")][string]$Action = "status")

$PgBin   = "D:\postgres\pgsql\bin"
$PgData  = "D:\postgres\data"
$PgLog   = "D:\postgres\postgres.log"
$PgAdmin = "D:\postgres\pgsql\pgAdmin 4\runtime\pgAdmin4.exe"

switch ($Action) {
    "start"   { & "$PgBin\pg_ctl.exe" -D $PgData -l $PgLog -o "-p 5432 -c listen_addresses=localhost" start }
    "stop"    { & "$PgBin\pg_ctl.exe" -D $PgData stop }
    "status"  { & "$PgBin\pg_ctl.exe" -D $PgData status }
    "psql"    { & "$PgBin\psql.exe" -h localhost -p 5432 -U postgres -d sentiment }
    "pgadmin" { Start-Process -FilePath $PgAdmin }
}
