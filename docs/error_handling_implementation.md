# Workflow Error Handling Implementation

## Overview

This document describes the comprehensive error handling system implemented for the Agent Jailbreaking Research workflow. The system provides robust error recovery, monitoring, and logging capabilities to ensure reliable experiment execution.

## Key Components

### 1. Error Classification System

#### ErrorType Enum
- `AGENT_ERROR`: Errors related to agent processing
- `DATABASE_ERROR`: Database connection and query errors
- `MODEL_ERROR`: LLM API and model-related errors
- `VALIDATION_ERROR`: Data validation and format errors
- `TIMEOUT_ERROR`: Operation timeout errors
- `NETWORK_ERROR`: Network connectivity issues
- `CONFIGURATION_ERROR`: System configuration problems
- `UNKNOWN_ERROR`: Unclassified errors

#### ErrorSeverity Enum
- `LOW`: Minor issues that don't affect core functionality
- `MEDIUM`: Moderate issues that may impact performance
- `HIGH`: Serious issues that affect experiment execution
- `CRITICAL`: Severe issues that prevent system operation

### 2. WorkflowError Exception Class

Enhanced exception class with metadata:
- Error type and severity classification
- Recoverability indication
- Contextual information
- Original exception preservation
- Timestamp tracking
- Structured logging support

```python
workflow_error = WorkflowError(
    message="Model API failed",
    error_type=ErrorType.MODEL_ERROR,
    severity=ErrorSeverity.HIGH,
    recoverable=True,
    context={"model": "gpt-4o", "attempt": 1},
    original_error=original_exception
)
```

### 3. Circuit Breaker Pattern

Prevents cascade failures by monitoring failure rates:
- **Closed State**: Normal operation, requests pass through
- **Open State**: Failures exceeded threshold, requests blocked
- **Half-Open State**: Testing recovery, limited requests allowed

Configuration:
- `failure_threshold`: Number of failures before opening (default: 5)
- `recovery_timeout`: Seconds to wait before attempting recovery (default: 60)

### 4. Workflow Monitoring

Comprehensive monitoring system tracking:
- Total experiments executed
- Success/failure rates
- Error frequency by type and severity
- Average execution duration
- Circuit breaker trip counts
- Active experiment tracking

### 5. Node-Level Error Handling

Each workflow node is wrapped with error handling that provides:
- Automatic error classification
- Timeout protection
- Circuit breaker integration
- Structured logging
- Retry coordination
- Context preservation

## Error Recovery Mechanisms

### 1. Retry Logic

Configurable retry system with:
- Maximum retry limits per operation
- Exponential backoff (future enhancement)
- Operation-specific retry counts
- Automatic retry count reset on success

### 2. Fallback Strategies

#### Model Error Recovery
- Primary/secondary model switching
- API key rotation (future enhancement)
- Rate limit backoff

#### Database Error Recovery
- Connection pool refresh
- Transaction retry
- Read replica fallback (future enhancement)

#### Network Error Recovery
- Automatic retry with delay
- Connection timeout adjustment
- DNS resolution retry

### 3. Graceful Degradation

When recovery fails:
- Partial result preservation
- Experiment state persistence
- Detailed error logging
- Clean resource cleanup

## Monitoring and Alerting

### Health Status Monitoring

The system provides comprehensive health status including:
- Overall system health (healthy/degraded/unhealthy)
- Success rate tracking
- Active experiment count
- Error distribution analysis
- Circuit breaker status
- Average execution duration

### Logging Integration

Structured logging with:
- Error context preservation
- Experiment correlation
- Performance metrics
- Circuit breaker events
- Recovery attempt tracking

## Configuration

### Timeout Configuration

Node-specific timeouts:
- `persuader_turn`: 60 seconds
- `persuadee_evaluation`: 60 seconds
- `safety_judge`: 45 seconds
- `generate_final_response`: 60 seconds
- `initialize_conversation`: 30 seconds
- Default: 30 seconds

### Circuit Breaker Configuration

Service-specific circuit breakers:
- `model_api`: For agent operations (threshold: 5, timeout: 60s)
- `database`: For database operations (threshold: 3, timeout: 30s)

### Retry Configuration

- Default maximum retries: 3
- Configurable per workflow instance
- Operation-specific retry tracking

## Usage Examples

### Basic Workflow Creation

```python
from src.workflow.graph import create_conversation_workflow

# Create workflow with custom retry limit
workflow = create_conversation_workflow(max_retries=5)

# Get health status
health = workflow.get_workflow_health()
print(f"System health: {health['health_status']}")
```

### Error Handling in Custom Nodes

```python
async def custom_node(state: ConversationState) -> ConversationState:
    try:
        # Your node logic here
        result = await some_operation(state)
        return result
    except Exception as e:
        # Error will be automatically handled by wrapper
        raise WorkflowError(
            message=f"Custom operation failed: {str(e)}",
            error_type=ErrorType.AGENT_ERROR,
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            context={"operation": "custom_operation"},
            original_error=e
        )
```

### Monitoring Integration

```python
# Configure logging
from src.workflow.graph import configure_workflow_logging

configure_workflow_logging(
    log_level="INFO",
    log_file="workflow_errors.log"
)

# Run workflow with monitoring
async with workflow.workflow_context(state):
    result = await workflow.run_conversation(state)
```

## Testing

The error handling system includes comprehensive tests:

### Unit Tests
- Error classification accuracy
- Severity determination logic
- Circuit breaker functionality
- Retry mechanism behavior
- Recovery strategy effectiveness

### Integration Tests
- End-to-end error handling flow
- Multi-error scenario handling
- Timeout and recovery testing
- Monitoring system validation
- Performance under error conditions

### Test Execution

```bash
# Run error handling tests
python -m pytest tests/test_workflow_error_handling.py -v

# Run simple integration test
python test_integration_simple.py

# Run basic functionality test
python test_error_handling_simple.py
```

## Performance Considerations

### Error Handling Overhead
- Minimal performance impact during normal operation
- Error classification is lightweight
- Circuit breaker checks are O(1)
- Monitoring updates are asynchronous

### Memory Management
- Error context is bounded in size
- Retry counts are automatically cleaned up
- Circuit breaker state is minimal
- Monitoring metrics use rolling windows

### Scalability
- Thread-safe error handling
- Concurrent experiment support
- Independent circuit breaker states
- Distributed monitoring ready (future)

## Future Enhancements

### Planned Improvements
1. **Advanced Retry Strategies**
   - Exponential backoff with jitter
   - Adaptive retry limits based on error type
   - Circuit breaker integration with retries

2. **Enhanced Monitoring**
   - Real-time dashboards
   - Alerting integration (email, Slack)
   - Metrics export (Prometheus, etc.)

3. **Recovery Strategies**
   - Automatic model switching
   - Database failover support
   - Experiment checkpointing and resume

4. **Configuration Management**
   - Dynamic configuration updates
   - Environment-specific settings
   - A/B testing for error handling strategies

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure proper Python path configuration
   - Check for circular import dependencies
   - Verify all required packages are installed

2. **Circuit Breaker Stuck Open**
   - Check failure threshold configuration
   - Verify recovery timeout settings
   - Use `workflow.reset_circuit_breakers()` if needed

3. **High Error Rates**
   - Review error classification accuracy
   - Check external service availability
   - Validate retry configuration
   - Monitor resource utilization

### Debugging Tools

```python
# Get detailed health status
health = workflow.get_workflow_health()
print(json.dumps(health, indent=2))

# Reset circuit breakers
workflow.reset_circuit_breakers()

# Check retry counts
print(workflow.flow_controller.retry_counts)
```

## Conclusion

The comprehensive error handling system provides robust protection against various failure modes while maintaining system performance and reliability. The combination of error classification, circuit breakers, retry logic, and monitoring ensures that the Agent Jailbreaking Research workflow can handle real-world operational challenges effectively.