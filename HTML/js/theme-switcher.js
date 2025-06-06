// 主题切换脚本

// 当DOM加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 从会话存储中获取主题设置
    const currentTheme = localStorage.getItem('theme') || 'raspberry_pi';
    
    // 应用主题
    applyTheme(currentTheme);
    
    // 监听主题选择变化
    const themeOptions = document.querySelectorAll('.theme-option');
    if (themeOptions.length > 0) {
        themeOptions.forEach(option => {
            option.addEventListener('click', function() {
                const theme = this.getAttribute('data-theme');
                if (theme) {
                    // 更新隐藏输入
                    document.getElementById('theme').value = theme;
                    
                    // 更新活动类
                    themeOptions.forEach(opt => {
                        opt.classList.remove('active');
                    });
                    this.classList.add('active');
                    
                    // 应用主题（立即预览）
                    applyTheme(theme);
                    
                    // 保存到本地存储（临时，直到保存设置）
                    localStorage.setItem('theme', theme);
                }
            });
        });
    }
});

// 应用主题函数
function applyTheme(theme) {
    // 移除所有主题属性
    document.documentElement.removeAttribute('data-theme');
    
    // 根据主题名称设置属性
    switch(theme) {
        case 'dark':
            document.documentElement.setAttribute('data-theme', 'dark');
            break;
        case 'fancy':
            document.documentElement.setAttribute('data-theme', 'fancy');
            break;
        case 'ocean':
            document.documentElement.setAttribute('data-theme', 'ocean');
            break;
        case 'raspberry_pi':
        default:
            // 默认主题不需要设置属性
            break;
    }
}
