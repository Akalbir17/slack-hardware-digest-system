-- Initialize Slack Digest System Database
-- This script creates the database and initial schema

-- Create database
CREATE DATABASE slack_digest;

-- Connect to the new database
\c slack_digest;

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slack_user_id VARCHAR(50) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    display_name VARCHAR(100),
    email VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create channels table
CREATE TABLE IF NOT EXISTS channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slack_channel_id VARCHAR(50) UNIQUE NOT NULL,
    channel_name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slack_message_id VARCHAR(50) UNIQUE NOT NULL,
    channel_id UUID REFERENCES channels(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    thread_ts VARCHAR(50),
    reply_count INTEGER DEFAULT 0,
    is_processed BOOLEAN DEFAULT false,
    sentiment_score FLOAT CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    priority_level VARCHAR(20) DEFAULT 'normal' CHECK (priority_level IN ('low', 'normal', 'high', 'critical')),
    tags TEXT[],
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create risk_assessments table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    risk_category VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    mitigation_suggestions TEXT[],
    requires_attention BOOLEAN DEFAULT false,
    assigned_to VARCHAR(100),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'dismissed')),
    resolution_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Create digests table for daily summaries
CREATE TABLE IF NOT EXISTS digests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    key_topics TEXT[],
    total_messages INTEGER DEFAULT 0,
    sentiment_avg FLOAT,
    risk_count INTEGER DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_is_processed ON messages(is_processed);
CREATE INDEX IF NOT EXISTS idx_messages_priority_level ON messages(priority_level);
CREATE INDEX IF NOT EXISTS idx_messages_sentiment_score ON messages(sentiment_score);

CREATE INDEX IF NOT EXISTS idx_risk_assessments_message_id ON risk_assessments(message_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_risk_level ON risk_assessments(risk_level);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_status ON risk_assessments(status);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_requires_attention ON risk_assessments(requires_attention);

CREATE INDEX IF NOT EXISTS idx_digests_date ON digests(date);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_channels_updated_at BEFORE UPDATE ON channels FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_messages_updated_at BEFORE UPDATE ON messages FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_risk_assessments_updated_at BEFORE UPDATE ON risk_assessments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_digests_updated_at BEFORE UPDATE ON digests FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data
INSERT INTO channels (slack_channel_id, channel_name, description) VALUES 
('C1234567890', 'general', 'General team discussion'),
('C1234567891', 'hardware-gtm', 'Hardware Go-To-Market discussions'),
('C1234567892', 'product-updates', 'Product update announcements'),
('C1234567893', 'engineering', 'Engineering team coordination'),
('C1234567894', 'sales-alerts', 'Sales team alerts and updates')
ON CONFLICT (slack_channel_id) DO NOTHING;

INSERT INTO users (slack_user_id, username, display_name, email) VALUES 
('U1234567890', 'alice.johnson', 'Alice Johnson', 'alice@company.com'),
('U1234567891', 'bob.smith', 'Bob Smith', 'bob@company.com'),
('U1234567892', 'charlie.brown', 'Charlie Brown', 'charlie@company.com'),
('U1234567893', 'diana.prince', 'Diana Prince', 'diana@company.com'),
('U1234567894', 'eve.wilson', 'Eve Wilson', 'eve@company.com')
ON CONFLICT (slack_user_id) DO NOTHING;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO slack_digest_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO slack_digest_user; 