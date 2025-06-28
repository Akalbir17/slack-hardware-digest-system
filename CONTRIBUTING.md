# Contributing to Slack Hardware Digest System

Thank you for your interest in contributing to the Slack Hardware Digest System! We welcome contributions from everyone and are grateful for even the smallest of fixes!

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git
- OpenAI API key for testing AI features

### Development Setup

1. **Fork the repository**
   ```bash
   git fork https://github.com/YOUR_USERNAME/slack-hardware-digest-system
   cd slack-hardware-digest-system
   ```

2. **Set up your development environment**
   ```bash
   # Copy environment template
   cp env.template .env
   
   # Add your OpenAI API key to .env
   # OPENAI_API_KEY=your-key-here
   
   # Start development services
   docker-compose up -d
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

## 🛠️ Development Workflow

### Branching Strategy
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/feature-name` - Individual features
- `bugfix/bug-description` - Bug fixes
- `hotfix/critical-fix` - Critical production fixes

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-amazing-feature
   ```

2. **Make your changes**
   - Write clean, readable code
   - Follow existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run the test suite
   python -m pytest tests/
   
   # Test the system end-to-end
   docker-compose up -d
   curl http://localhost:8000/health
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

## 📝 Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Examples:**
```
feat: add new risk assessment algorithm
fix: resolve agent confidence score calculation
docs: update API documentation for new endpoints
refactor: simplify message processing pipeline
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/

# Run specific test files
python -m pytest tests/test_agents.py
```

### Test Coverage
- Aim for >80% code coverage
- Write unit tests for all new functions
- Add integration tests for API endpoints
- Include end-to-end tests for critical workflows

### Manual Testing
1. Start the system: `docker-compose up -d`
2. Access dashboard: http://localhost:8501
3. Test all major features
4. Verify AI agents are working
5. Check API endpoints

## 📊 Code Quality

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write descriptive variable and function names
- Keep functions small and focused

### Pre-commit Hooks
We recommend setting up pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

### Static Analysis
```bash
# Code formatting
black .

# Import sorting
isort .

# Type checking
mypy app/

# Linting
flake8 .
```

## 🤖 AI Agent Development

### Adding New Agents

1. **Create agent class** in `agents/` directory
2. **Inherit from BaseAgent**
3. **Implement required methods:**
   - `analyze_messages()`
   - `get_system_prompt()`
   
4. **Register agent** in `agent_manager.py`
5. **Add tests** for the new agent
6. **Update documentation**

### Agent Guidelines
- Each agent should have a specific focus area
- Use clear, descriptive system prompts
- Implement proper error handling
- Return structured data in AgentResponse format
- Include confidence scores and metadata

## 📚 Documentation

### API Documentation
- Document all new endpoints
- Include request/response examples
- Update OpenAPI/Swagger specs
- Add endpoint descriptions

### Code Documentation
- Use docstrings for all functions and classes
- Include parameter and return type descriptions
- Add usage examples where helpful
- Keep documentation up to date

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description** - Clear description of the issue
2. **Steps to Reproduce** - Detailed steps to reproduce the bug
3. **Expected Behavior** - What you expected to happen
4. **Actual Behavior** - What actually happened
5. **Environment** - OS, Python version, Docker version
6. **Logs** - Relevant log output or error messages

## 💡 Feature Requests

For feature requests, please provide:

1. **Problem Statement** - What problem does this solve?
2. **Proposed Solution** - How should it work?
3. **Alternatives** - Other solutions you considered
4. **Additional Context** - Any other relevant information

## 🔄 Pull Request Process

1. **Create Pull Request**
   - Use descriptive title and description
   - Reference related issues
   - Include testing instructions

2. **Code Review**
   - All PRs require at least one review
   - Address feedback promptly
   - Keep discussions constructive

3. **Merge Requirements**
   - All tests must pass
   - Code coverage should not decrease
   - Documentation must be updated
   - No merge conflicts

## 🏷️ Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `MAJOR` - Breaking changes
- `MINOR` - New features (backward compatible)
- `PATCH` - Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Create release notes
- [ ] Tag the release
- [ ] Deploy to production

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- 🧪 **Testing** - Expand test coverage
- 📚 **Documentation** - Improve guides and examples
- 🤖 **AI Agents** - New specialized agents
- 🔧 **Performance** - Optimization and scaling

### Medium Priority
- 🎨 **UI/UX** - Dashboard improvements
- 🔌 **Integrations** - New platform connections
- 📊 **Analytics** - Advanced metrics and insights
- 🚀 **DevOps** - Deployment and monitoring tools

### Good First Issues
Look for issues labeled `good-first-issue` for beginner-friendly tasks:
- Documentation improvements
- Small bug fixes
- Test additions
- Code cleanup

## ❓ Questions?

- 📧 Open an issue for questions
- 💬 Join our discussions
- 📖 Check the documentation
- 🤝 Reach out to maintainers

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to making hardware GTM teams more efficient! 🚀 