from kafka import KafkaConsumer
import psycopg2
import json

# =====================================================
# PostgreSQL Connection
# =====================================================

conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="postgres123"
)

cursor = conn.cursor()

# =====================================================
# Kafka Consumer
# =====================================================

consumer = KafkaConsumer(
    "weather_topic",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="weather-group",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("=" * 60)
print("Weather Consumer Started...")
print("=" * 60)

# =====================================================
# Consume Messages
# =====================================================

for message in consumer:

    try:

        data = message.value

        print("\nReceived Weather Data")
        print(data)

        current = data.get("current", {})

        weather_time = current.get("time")
        temperature = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind_speed = current.get("wind_speed_10m")

        # Skip incomplete records
        if (
            weather_time is None
            or temperature is None
            or humidity is None
            or wind_speed is None
        ):
            print("Incomplete record skipped.")
            continue

        # =============================================
        # Duplicate Check
        # =============================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM weather_data
            WHERE time=%s
            """,
            (weather_time,)
        )

        exists = cursor.fetchone()[0]

        if exists > 0:
            print("Duplicate record skipped.")
            continue

        # =============================================
        # Insert into PostgreSQL
        # =============================================

        cursor.execute(
            """
            INSERT INTO weather_data
            (
                time,
                temperature_2m,
                relative_humidity_2m,
                wind_speed_10m
            )
            VALUES
            (
                %s,%s,%s,%s
            )
            """,
            (
                weather_time,
                temperature,
                humidity,
                wind_speed
            )
        )

        conn.commit()

        print("Inserted Successfully")

    except Exception as e:

        conn.rollback()

        print("Database Error :", e)

# =====================================================
# Close Connection
# =====================================================

consumer.close()
cursor.close()
conn.close()