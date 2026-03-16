use crate::{BoxFuture, Error, KvPairs, Prefix, Storage};
use rusqlite::Connection;
use std::path::Path;
use tokio::sync::Mutex;

pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let conn =
            Connection::open(path).map_err(|e| Error::Internal(format!("sqlite open: {e}")))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS kv (key BLOB PRIMARY KEY, value BLOB NOT NULL);
             CREATE TABLE IF NOT EXISTS counters (key BLOB PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);",
        )
        .map_err(|e| Error::Internal(format!("sqlite init: {e}")))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

impl Storage for SqliteStorage {
    fn get(&self, key: &[u8]) -> BoxFuture<'_, Result<Option<Vec<u8>>, Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            let conn = self.conn.lock().await;
            let mut stmt = conn
                .prepare_cached("SELECT value FROM kv WHERE key = ?")
                .map_err(|e| Error::Internal(e.to_string()))?;
            let result = stmt.query_row([&key], |row| row.get::<_, Vec<u8>>(0)).ok();
            Ok(result)
        })
    }

    fn set(&self, key: &[u8], value: Vec<u8>) -> BoxFuture<'_, Result<(), Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            let conn = self.conn.lock().await;
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params![key, value],
            )
            .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(())
        })
    }

    fn increment(&self, key: &[u8], delta: i64) -> BoxFuture<'_, Result<i64, Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            let conn = self.conn.lock().await;
            let value: i64 = conn
                .query_row(
                    "INSERT INTO counters (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = value + excluded.value RETURNING value",
                    rusqlite::params![key, delta],
                    |row| row.get(0),
                )
                .map_err(|e| Error::Internal(e.to_string()))?;

            Ok(value)
        })
    }

    fn list(&self, prefix: &Prefix) -> BoxFuture<'_, Result<KvPairs, Error>> {
        let prefix_vec = prefix.to_vec();
        let mut upper = prefix_vec.clone();
        // Increment last byte for upper bound of range scan.
        if let Some(last) = upper.last_mut() {
            *last = last.wrapping_add(1);
        }

        Box::pin(async move {
            let conn = self.conn.lock().await;

            let mut result = Vec::new();

            // Collect from kv table.
            let mut stmt = conn
                .prepare_cached("SELECT key, value FROM kv WHERE key >= ? AND key < ?")
                .map_err(|e| Error::Internal(e.to_string()))?;
            let kv_rows = stmt
                .query_map(rusqlite::params![prefix_vec, upper], |row| {
                    Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
                })
                .map_err(|e| Error::Internal(e.to_string()))?;

            for pair in kv_rows.flatten() {
                result.push(pair);
            }

            // Also collect from counters table (budget/usage keys live here).
            let mut stmt = conn
                .prepare_cached("SELECT key, value FROM counters WHERE key >= ? AND key < ?")
                .map_err(|e| Error::Internal(e.to_string()))?;
            let counter_rows = stmt
                .query_map(rusqlite::params![prefix_vec, upper], |row| {
                    let k: Vec<u8> = row.get(0)?;
                    let v: i64 = row.get(1)?;
                    Ok((k, v.to_le_bytes().to_vec()))
                })
                .map_err(|e| Error::Internal(e.to_string()))?;

            for pair in counter_rows.flatten() {
                if !result.iter().any(|(k, _)| k == &pair.0) {
                    result.push(pair);
                }
            }

            Ok(result)
        })
    }

    fn delete(&self, key: &[u8]) -> BoxFuture<'_, Result<(), Error>> {
        let key = key.to_vec();
        Box::pin(async move {
            let conn = self.conn.lock().await;
            conn.execute("DELETE FROM kv WHERE key = ?", [&key])
                .map_err(|e| Error::Internal(e.to_string()))?;
            conn.execute("DELETE FROM counters WHERE key = ?", [&key])
                .map_err(|e| Error::Internal(e.to_string()))?;
            Ok(())
        })
    }
}
